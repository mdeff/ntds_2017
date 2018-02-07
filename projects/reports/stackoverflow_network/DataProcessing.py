import json
import xml.etree.ElementTree as et
from lxml import etree
import os
import ijson
from dateutil import parser
import datetime

def split_qa_json(questions_path, answers_path, posts_path):
    """
    Creates a JSON file each for questions and answers containing only selected features:
        id, owner, time, votes, tags and reference
    """
    with open(questions_path, 'w') as q, open(answers_path, 'w') as a:
        q.write("{")
        a.write("{")
        
        # don't add , before first entry
        firstQ = True
        firstA = True
        for event, elem in et.iterparse(posts_path):
            if elem.tag == "row":
                # Fetch id, owner, time, votes, tags and reference
                try:
                    parsed = {}
                    parsed["id"] = elem.attrib["Id"]
                    parsed["owner_id"] = elem.attrib["OwnerUserId"]
                    parsed["time"] = elem.attrib["CreationDate"]
                    parsed["votes"] = elem.attrib["Score"]
                    
                    try:                    
                        parsed["tags"] = elem.attrib["Tags"]
                    except KeyError:
                        parsed["tags"] = ""
                        
                    if elem.attrib["PostTypeId"] == "1":
                        #Question
                        parsed["ref"] = elem.attrib["AcceptedAnswerId"]
                        if not firstQ:
                            q.write(",\n")
                        else:
                            firstQ = False
                        q.write( "\"" + parsed["id"] + "\":" + json.dumps(parsed))
                    else :
                        # Answer
                        parsed["ref"] = elem.attrib["ParentId"]
                        if not firstA:
                            a.write(",\n")
                        else:
                            firstA = False
                        a.write( "\"" + parsed["ref"] + "\":" + json.dumps(parsed))
                except KeyError as e:
                    pass
                    # ignore posts without answer for now
                    # ignore posts without user id
            elem.clear()
        q.write("}")
        a.write("}")
            
def create_edge_list(questions_path, answers_path, edge_list_path):
    """
    Create edge list based on question and answer file and save it
    """
    with open(questions_path, 'r') as q, open(answers_path, 'r') as a, open(edge_list_path,'w') as e:
        questions = json.load(q)
        answers = json.load(a)
        e.write("{ \"edges\":[")
        first = True
        # find matching accepted answer
        for question in questions:
            if question in answers:
                answer = answers[question]
                if not first:
                    e.write(",\n")
                else:
                    first = False
                e.write(json.dumps({'q_id':questions[question]["owner_id"], 'a_id':answer["owner_id"], 'time':answer["time"], 'tags':questions[question]["tags"], "votes":answer["votes"]}))
        e.write("]}")
            
def split_qa_json_all(questions_path, answers_path, posts_path):
    """
    Creates a JSON file each for questions and answers containing only selected features:
        id, owner, time, votes, tags and reference
    """
       
    with open(questions_path, 'w') as q, open(answers_path, 'w') as a:
        q.write("{")
        a.write("{")
        
        # don't add , before first entry
        firstQ = True
        firstA = True
        for event, elem in etree.iterparse(posts_path):
            if elem.tag == "row":
                # Fetch id, owner, time, votes, tags and reference
                try:
                    parsed = {}
                    parsed["id"] = elem.attrib["Id"]
                    parsed["owner_id"] = elem.attrib["OwnerUserId"]
                    parsed["time"] = elem.attrib["CreationDate"]
                    parsed["votes"] = elem.attrib["Score"]
                    
                    try:                    
                        parsed["tags"] = elem.attrib["Tags"]
                    except KeyError:
                        parsed["tags"] = ""
                        
                    if elem.attrib["PostTypeId"] == "1":
                        #Question
                        parsed["ref"] = elem.attrib["AcceptedAnswerId"]
                        if not firstQ:
                            q.write(",\n")
                        else:
                            firstQ = False
                        q.write( "\"" + parsed["id"] + "\":" + json.dumps(parsed))
                    else :
                        # Answer
                        parsed["ref"] = elem.attrib["ParentId"]
                        if not firstA:
                            a.write(",\n")
                        else:
                            firstA = False
                        a.write( "\"" + parsed["ref"]+ "_" + parsed["id"] + "\":" + json.dumps(parsed))
                except KeyError as e:
                    pass
                    # ignore posts without answer for now
                    # ignore posts without user id
            elem.clear()
        q.write("}")
        a.write("}")
            
def create_edge_list_all(questions_path, answers_path, edge_list_path):
    """
    Create edge list based on question and answer file and save it
    """
    with open(questions_path, 'r') as q, open(answers_path, 'r') as a, open(edge_list_path,'w') as e:
        questions = json.load(q)
        answers = json.load(a)
        e.write("{ \"edges\":[")
        first = True
        # find matching accepted answer
        for answerkey in answers:
            question = answerkey.split("_")[0]
            if question in questions:
                answer = answers[answerkey]
                if not first:
                    e.write(",\n")
                else:
                    first = False
                e.write(json.dumps({'q_id':questions[question]["owner_id"], 'a_id':answer["owner_id"], 'time':answer["time"], 'tags':questions[question]["tags"], "votes_a":answer["votes"], "votes_q":questions[question]["votes"], "accepted":answer["id"]==questions[question]["ref"]}))
        e.write("]}")
            
def split_edge_list_tags(edge_list_tag_path, edge_list_path):
    """
    Split the edge list to several edge lists using tags
    """
    
    # delete old files
    clear_edge_lists_tags(edge_list_tag_path)    
    
    # read edge list
    with open(edge_list_path,'r') as e:
        edges = ijson.items(e, "edges.item")
        
        # handle each edge
        for edge in edges:
            
            # split tags
            tags = [s.replace("<", "").replace(">","") for s in edge["tags"].split("><")]
            edge.pop('tags', None)
            
            for tag in tags:
            
                # Write to correspodning files
                with open(os.path.join(edge_list_tag_path, "{}.json".format(tag)), 'a') as e_tag:
                    e_tag.write(json.dumps(edge)+"\n")
    
    # format files
    format_edge_lists_tags(edge_list_tag_path)
                    
def clear_edge_lists_tags(edge_list_tag_path):
    """
    Removes previously saved edge lists
    """
    for the_file in os.listdir(edge_list_tag_path):
        file_path = os.path.join(edge_list_tag_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
            
def format_edge_lists_tags(edge_list_tag_path):
    """
    Formats the edge lists as proper json
    """
    for the_file in os.listdir(edge_list_tag_path):
        file_path = os.path.join(edge_list_tag_path, the_file)
        with open(file_path,'r+') as e:
            content = e.read()
            content = content.replace("}\n{", "},\n{")
            e.seek(0, 0)
            e.write( "{ \"edges\":[" + '\n' + content)
            e.write("]}")
            
def order_edge_lists_tags_time(edge_list_tag_path):
    """
    Create new edge list with edges ordered by time
    """
    
    for the_file in os.listdir(edge_list_tag_path):
        if "_ordered" in the_file or "_list" in the_file:
            continue
        file_path = os.path.join(edge_list_tag_path, the_file)
        file_ordered_path = os.path.join(edge_list_tag_path, the_file.replace(".json", "_ordered.json"))
        with open(file_path,'r') as e, open(file_ordered_path,'w') as e_ordered:
            edges = json.load(e)
            edges["edges"].sort(key=lambda elem: parser.parse(elem["time"]))
            for edge in edges["edges"]:
                edge['time'] = unix_time_millis(parser.parse(edge["time"]))
            e_ordered.write(json.dumps(edges))

epoch = datetime.datetime.utcfromtimestamp(0)
def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0
            
def edge_lists_to_txt(edge_list_tag_path):
    """
    Converts the ordered edge lists to txt files
    """
    for the_file in os.listdir(edge_list_tag_path):
        if not "_ordered.json" in the_file:
            continue
        file_path = os.path.join(edge_list_tag_path, the_file)
        list_path = os.path.join(edge_list_tag_path, the_file.replace(".json", "_list.txt"))
        with open(file_path,'r') as e, open(list_path,'w') as l:
            edges = ijson.items(e, "edges.item")
            for edge in edges:
                l.write(edge["q_id"]+ " " + edge["a_id"]+ " "+ str(edge["time"])+ " "+ edge["votes_q"]+ " "+ edge["votes_a"] +" "+ str(edge["accepted"]) + "\n")