import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import time, sleep
import os
import json

from settings import AWS_FOLDER, CLIP_INFO_FILE


def is_description_tag(tag):
    if tag.has_attr('face') and tag.has_attr('size') and tag['face'] == 'arial' and tag['size'] == '2':
        return True
    else:
        return False


def handle_url(url):
    try:
        r = requests.get(url)
    except requests.exceptions.ConnectionError:
        print('connection error')
        sleep(5)
        r =requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    tag = soup.find(is_description_tag)
    descriptions_parts = []
    if tag is not None:
        for child in list(tag.strings)[7:]:
            item = child.replace("\n", " ").strip()
            if (item) == "":
                break
            else:
                descriptions_parts.append(item)
    if descriptions_parts:
        description = (" ".join(descriptions_parts))
        return(description)
    else:
        return None


if __name__ == '__main__':
    url_to_desc = {}
    db = pd.read_csv(CLIP_INFO_FILE, sep="\t")
    urls = set(db['url'])
    start = time()
    gotten = 0
    for i, url in enumerate(urls):
        output = handle_url(url)
        if output:
            gotten += 1
            url_to_desc[url] = output
    print(time() - start)
    print(gotten)
    print(len(urls))
    json.dump(url_to_desc, open(os.path.join(AWS_FOLDER, 'descriptions.json'), 'w'), ensure_ascii=False, indent=2)
