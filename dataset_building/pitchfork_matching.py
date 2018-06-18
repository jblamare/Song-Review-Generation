import sqlite3
import sys
import os
import time
import datetime
import pandas
import pickle as cP
from settings import MSD_MP3_FOLDER, MSD_SONGS_FOLDER, MSD_DATABASE_FOLDER, MSD_CODE_FOLDER, MSD_SPLIT_FOLDER, PITCHFORK_DB_PATH, PITCHFORK_CSV_PATH
import json

sys.path.append(os.path.join(MSD_CODE_FOLDER, 'PythonSrc'))

import hdf5_getters as GETTERS


def encode_string(s):
    """
    Simple utility function to make sure a string is proper
    to be used in a SQLite query
    (different than posgtresql, no N to specify unicode)
    EXAMPLE:
      That's my boy! -> 'That''s my boy!'
    """
    s = str(s).lower()
    # s = " ".join(w.capitalize() for w in s.split())
    return "'"+s.replace("'", "''")+"'"


def encode_string_like(s):
    s = str(s).lower()
    # s = " ".join(w.capitalize() for w in s.split())
    return "'%"+s.replace("'", "''")+"%'"


def encode_list(l):
    return "("+", ".join([encode_string(a) for a in l])+")"


def strtimedelta(starttime, stoptime):
    return str(datetime.timedelta(seconds=stoptime-starttime))


conn_msd = sqlite3.connect(os.path.join(MSD_DATABASE_FOLDER, 'track_metadata.db'))
c_msd = conn_msd.cursor()
conn_pf = sqlite3.connect(PITCHFORK_DB_PATH)
c_pf = conn_pf.cursor()


def create_pairing_from_msd():
    ######### 1) get pairs from MSD then check in PF #########

    # Retrieve all artist, album pairs from the MSD
    q_msd_pairs = "SELECT track_id, title, release, artist_name FROM songs GROUP BY release, artist_name"
    res_msd_pairs = conn_msd.execute(q_msd_pairs)
    pairs = res_msd_pairs.fetchall()

    n_reviews = 0
    n_artists = 0
    f = open('pitchfork_msd.csv', 'w')
    f.write('track_id,title,album,artist,reviewid'+'\n')
    for track_id, title, release, artist in pairs:
        if n_reviews % 1000 == 0:
            print(n_reviews)
            print(n_artists)
            print('-------')
        # Retrieve separate reviewids for this artist
        q_pf_artists = "SELECT reviewid FROM artists"
        q_pf_artists += " WHERE artist="+encode_string(artist)
        res_pf_artists = conn_pf.execute(q_pf_artists)
        reviewid_list = [reviewid[0] for reviewid in res_pf_artists.fetchall()]
        reviewid_list = encode_list(reviewid_list)

        q_pf_reviews = "SELECT reviewid, title, artist FROM reviews"
        q_pf_reviews += " WHERE title="+encode_string(release)
        q_pf_reviews += " AND artist LIKE "+encode_string_like(artist)
        q_pf_reviews += " AND reviewid IN "+reviewid_list
        # t1 = time.time()
        res_pf_reviews = conn_pf.execute(q_pf_reviews)
        # t2 = time.time()
        # print("Request time = ", strtimedelta(t1, t2))
        reviews = res_pf_reviews.fetchall()
        if len(reviews) > 0:
            n_artists += 1
            for reviewid, _, _ in reviews:
                f.write(str(track_id)+','+title+','+release+','+artist+','+str(reviewid)+'\n')

        n_reviews += 1

    f.close()
    print(n_reviews)
    print(n_artists)


def create_pairing_from_pf():
    ######### 2) get pairs from PF then check in MSD #########

    # Retrieve all pitchfork reviews
    q_pf = "SELECT reviewid, title FROM reviews"
    res_pf = conn_pf.execute(q_pf)
    reviews = res_pf.fetchall()

    n_reviews = 0
    n_artists = 0
    for reviewid, album in reviews:
        if n_reviews % 10 == 0:
            print(n_reviews)
            print(n_artists)
            print('-------')
        # Retrieve separate artists for the review
        q_pf_artists = "SELECT artist FROM artists WHERE reviewid="+str(reviewid)
        res_pf_artists = conn_pf.execute(q_pf_artists)
        artist_list = [artist[0] for artist in res_pf_artists.fetchall()]
        artist_list = encode_list(artist_list)

        q_msd = "SELECT release FROM songs"
        # q_msd += " WHERE lower(release)="+encode_string(album)
        q_msd += " WHERE lower(artist_name) IN "+artist_list
        # print(q_msd)
        # t1 = time.time()
        res_msd = conn_msd.execute(q_msd)
        # t2 = time.time()
        # print("Request time = ", strtimedelta(t1, t2))
        if len(res_msd.fetchall()) > 0:
            n_artists += 1
        n_reviews += 1

    print(n_reviews)


def create_pairing_from_pf_new_pitchfork():

    pitchfork = pandas.read_csv(open(PITCHFORK_CSV_PATH, encoding='cp1252'))

    n_reviews = 0
    n_matches = 0
    f = open('pitchfork_msd.csv', 'w')
    f.write('track_id,title,album,artist,review'+'\n')

    for index, row in pitchfork.iterrows():
        if n_reviews % 1000 == 0:
            print(n_reviews)
            print(n_matches)
            print('-------')

        pf_artist = str(row['artist'])
        pf_album = str(row['album'])

        q_msd_pairs = "SELECT track_id, title, release, artist_name FROM songs"
        q_msd_pairs += " WHERE release="+encode_string(pf_album)
        q_msd_pairs += " AND artist_name="+encode_string(pf_artist)
        res_msd_pairs = conn_msd.execute(q_msd_pairs)
        songs = res_msd_pairs.fetchall()

        if len(songs) > 0:
            n_matches += 1
            for track_id, title, _, _ in songs:
                f.write(str(track_id)+','+title+','+pf_album+','+pf_artist+','+row['review']+'\n')

        n_reviews += 1

    f.close()
    print(n_reviews)
    print(n_matches)


def create_pairing_from_msd_new_pitchfork():
    ######### 1) get pairs from MSD then check in PF #########

    print("RAS")

    # Retrieve all artist, album pairs from the MSD
    q_msd_pairs = "SELECT track_id, title, release, artist_name FROM songs GROUP BY release, artist_name"
    res_msd_pairs = conn_msd.execute(q_msd_pairs)
    pairs_msd = res_msd_pairs.fetchall()

    pitchfork = json.load(open('review_dictionary.json'))
    idmsd_to_tag = cP.load(open(MSD_SPLIT_FOLDER+'msd_id_to_tag_vector.cP', 'br'))

    n_reviews = 0
    n_matches = 0

    pairs = dict()
    for track_id, title, release, artist in pairs_msd:
        if n_reviews % 1000 == 0:
            print(n_reviews)
            print(n_matches)
            print('-------')

        n_reviews += 1

        try:
            review = pitchfork[artist.lower()][release.lower()]
            tags = idmsd_to_tag[track_id]
            n_matches += 1
            pair = dict()
            pair['title'] = title
            pair['album'] = release
            pair['artist'] = artist
            pair['review'] = review
            pairs[track_id] = pair
        except KeyError:
            continue

    json.dump(pairs, open(os.path.join(MSD_SPLIT_FOLDER, 'pairs.json'), 'w'))
    print(n_reviews)
    print(n_matches)


def check_pairing_reviews():
    pairing = pandas.read_csv('pitchfork_msd.csv', sep=',', header=0)
    # test = pairing.iloc[1]
    # reviewid = test['reviewid']
    # q_pf_review = "SELECT content FROM content WHERE reviewid="+encode_string(reviewid)
    # res_pf_review = conn_pf.execute(q_pf_review)
    # review = res_pf_review.fetchone()[0]
    # print(test)
    # print(review)
    idmsd_to_tag = cP.load(open(MSD_SPLIT_FOLDER+'msd_id_to_tag_vector.cP', 'br'))
    alright = 0
    indices = []
    for index, row in pairing.iterrows():
        try:
            tags = idmsd_to_tag[row['track_id']]
            alright += 1
            indices.append(index)
        except KeyError:
            pass
    pairing.iloc[indices].to_csv('pitchfork_msd_long.csv', index=False)
    print(alright)


# create_pairing_from_msd()
# create_pairing_from_pf()
create_pairing_from_msd_new_pitchfork()
# check_pairing_reviews()

conn_msd.close()
conn_pf.close()
