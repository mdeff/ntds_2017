import community
import pprint
import boto3 as boto3
from py2neo import Graph, Node, Relationship
import time
import networkx as nx



class Database(object):

    def __init__(self):
        self.graph = Graph("http://neo4j:qqwweerr@35.158.233.131:7474/db/data/")

    def get_orders(self, community_id):
        query = """
        MATCH 
        p = (a:Order) - [:PROCESSED_BY] - (:Community {name: '%s'})
        RETURN a.name AS name, a.algorithm AS algorithm, a.timestamp AS timestamp
        """ % (community_id,)

        return self.graph.run(query).data()

    def get_communities(self, order_id):
        query = """
        
        MATCH 
        p = (a:Order {name: '%s'}) - [:CREATE] - (c:Community)
        RETURN c.name AS name
        
        """ % (order_id,)

        return [i['name'] for i in self.graph.run(query).data()]

    def get_visited(self, community_id):
        query = """
        MATCH
            (a:Instauser) - [:MEMBER_OF] -> (:Community {name: %s}),
            (a)-[p:VISITED]-(b:Location)    
        RETURN  a.name AS Instauser, b.name AS LocationId, b.placeName AS placeName
        """ % (community_id, )

        data = self.graph.run(query).data()

        return data

    def get_order_with_partition(self, order_id):
        query = """
                 MATCH
            (:Order {name: %s}) - [:PROCESSED_BY] - (c:Community) ,
            (a:Instauser) - [:MEMBER_OF] -> (c),
            (b:Instauser) - [:MEMBER_OF] -> (c),

            (a)-[p:LIKED]-(b)

                RETURN  a.name AS source, b.name AS target
                """ % (order_id, )

        data = self.graph.run(query).data()

        G = nx.Graph()
        for i in data:
            G.add_node(i['source'])
            G.add_node(i['target'])
            G.add_edge(i['source'], i['target'])

        query = """
            MATCH
                p=(:Order {name: %s}) - [:CREATE] - (a:Community),
                (b:Instauser) - [:MEMBER_OF] - (a)
            RETURN  b.name AS instauser, a.name AS partition
            """ % (order_id, )

        data = self.graph.run(query).data()

        partitions = {}

        for i in data:
            partitions[i['instauser']] = i['partition']

        return G, partitions

    def get_community(self, community_id):
        query = """
         MATCH
    
    (a:Instauser) - [:MEMBER_OF] -> (:Community {name: %s}),
    (b:Instauser) - [:MEMBER_OF] -> (:Community {name: %s}),
    
    (a)-[p:LIKED]-(b)
    
        RETURN  a.name AS source, b.name AS target
        """ % (community_id, community_id)

        data = self.graph.run(query).data()
        G = nx.Graph()

        for i in data:
            G.add_node(i['source'])
            G.add_node(i['target'])
            G.add_edge(i['source'], i['target'])

        return G

    def insert_community_analysis(
            self,
            communities,
            parent_community_id,
            algorithm
    ):
        timestamp = int(float(time.time()))

        query = \
            """ 
            MATCH (a:Order) RETURN MAX(toInt(a.name)) AS m
        """

        max_result = self.graph.run(query).data()[0]

        max_id = max_result['m'] if max_result['m'] else 0
        order_id = int(float(max_id)) + 1

        query = \
            """ 
            CREATE (a:Order {name: %s, algorithm:'%s', timestamp:'%s'})
            RETURN a.name AS name
        """ % (order_id, str(algorithm), str(timestamp))
        name = self.graph.run(query).data()[0]['name']

        query = """
                    MATCH (a:Community),(b:Order)
                    WHERE a.name = {} AND b.name = {}
                    CREATE (a)-[r:PROCESSED_BY]->(b)
                    RETURN r
                """.format(parent_community_id, str(order_id))
        self.graph.run(query)

        query = \
            """ 
            MATCH (a:Community) RETURN MAX(toInt(a.name)) AS m
        """

        max_result = self.graph.run(query).data()[0]

        max_id = max_result['m'] if max_result['m'] else 0
        max_community = int(float(max_id)) + 1

        partitions_ids_map = {x: x + max_community for x in list(set(communities.values()))}
        for instauser in list(communities.keys()):
            communities[instauser] = partitions_ids_map[communities[instauser]]

        community_nodes = {}
        for community in list(set(partitions_ids_map.values())):
            query = """
                CREATE(n: Community {name: %s})
            """ % (community, )
            self.graph.run(query)

            query = """
                    MATCH (a:Community),(b:Community)
                    WHERE a.name = %s AND b.name = %s
                    CREATE (a)-[r:SUB_COMMUNITY_OF {order: %s}]->(b)
                    RETURN r
                """ % (community, parent_community_id, str(name))
            self.graph.run(query)

            query = """
                    MATCH (a:Order),(b:Community)
                    WHERE a.name = {} AND b.name = {}
                    CREATE (a)-[r:CREATE]->(b)
                    RETURN r
                """.format(order_id, community)
            self.graph.run(query)

        for instauser_id, partition in communities.items():
            query = """
                MATCH (a:Instauser),(b:Community)
                WHERE a.name = {} AND b.name = {}
                CREATE (a)-[r:MEMBER_OF]->(b)
                RETURN r
            """.format(instauser_id, partition)
            self.graph.run(query)

        return name

    def delete_community_analysis(
            self,
            order_id
    ):
        query = """
        MATCH 
            process_rel=(:Community) - [process_by:PROCESSED_BY] - (order:Order {name: '%s'}),
            create_rel=(order) - [create_:CREATE] - (community:Community), 
            sub_com_rel=(:Community) - [sub_community_of:SUB_COMMUNITY_OF] - (community), 
            member_rel=(:Instauser) - [member_of:MEMBER_OF] - (community)
            DELETE member_of, create_, sub_community_of, process_by, community, order
        """ % order_id
        self.graph.run(query)


if __name__ == "__main__":
    import community
    import pprint

    db = Database()

    community_id = 17

    graph, partitions = db.get_order_with_partition(community_id)

    pprint.pprint(partitions)
    #
    # network = db.get_community(community_id)
    # partitions = community.best_partition(network)
    #
    # parent_community_id = community_id
    # communities = partitions
    # algorithm = 'test'
    #
    # order_id = db.insert_community_analysis(communities, parent_community_id, algorithm)

    # algorithm = 'louvain'
    # community_id = '0'
    #
    # db = Database()
    # order = db.get_orders(community_id)[0]
    # communities = db.get_communities(order['name'])
    #
    # for community_id in communities:
    #
    #     order = db.get_orders(community_id)[0]
    #     communities2 = db.get_communities(order['name'])
    #
    #     for community_id_2 in communities2:
    #         parent_community_id = community_id_2
    #
    #         graph = db.get_community(community_id_2)
    #         partitions = community.best_partition(graph)
    #
    #         communities = partitions
    #         db.insert_community_analysis(communities, parent_community_id, algorithm)
    # graph = db.get_community(community_id)
    #
    # partitions = community.best_partition(graph)
    #
    # parent_community_id = community_id
    # communities = partitions
    # algorithm = 'louvain'
    #
    # db.insert_community_analysis(communities, parent_community_id, algorithm)
