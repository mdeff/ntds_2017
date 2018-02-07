import requests
from urllib.parse import quote

# Functions to fetch the data.
api_key = 'ca56ad16374050ce3da483cdaa8dbddd'

def parse_name(artist_name):
    return artist_name.replace(" ", "+")

def get_artist(artist_name):
    artist_name = parse_name(artist_name)
    print(artist_name)
    try:
        req = requests.get('http://ws.audioscrobbler.com/2.0/?method=artist.search&artist='+artist_name+'&api_key='+api_key+'&format=json')
        return req
    except:
        print("error above!")


def get_similar(artist_mbid):
    try:
        req = requests.get('http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&mbid='+artist_mbid+'&api_key='+api_key+'&format=json')
        return req
    except:
        print("error above!")

def parse_similar(req):
    try:
        temp_json = req.json()
        return temp_json['similarartists']['artist']
    except:
        print("error above!")

def parse_response(req, correct_url):
    try:
        temp_json = req.json()

        artists = temp_json['results']['artistmatches']['artist']
        for artist in artists:
            if (artist['url'][6:].lower()) == correct_url[5:].lower():
                return artist['mbid']
        return 'notfound'
    except:
        print("error above ")

def parse_artists(row):
    req = get_artist(row['name'])
    return parse_response(req, row.url)

def similar_artists(row):
    print(row['name'])
    req = get_similar(row['mbid'])
    return parse_similar(req)
