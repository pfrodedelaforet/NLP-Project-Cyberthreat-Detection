"""from selectorlib import Extractor
import requests
import json
url = 'https://www.bleepingcomputer.com/news/security/new-wapdropper-malware-stealthily-subscribes-you-to-premium-services/'
# Créer un extractor à partir de la elcture du fichier YAML
e = Extractor.from_yaml_file('yml_templates/text_bleeping.yml')
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
headers = {'User-Agent': user_agent}
# Télécharger la page en utilisant requests
r = requests.get(url, headers=headers)
# Passe le HTML de la page et créer
data = e.extract(r.text)
# Imprimer les données
print(json.dumps(data, indent=True))"""
from selectorlib import Extractor
import requests
import json
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from newsplease import NewsPlease
import re
e = Extractor.from_yaml_file('../yml_templates/text_nakedsec.yml')
user_agent = 'projet recherche datasophia'
headers = {'User-Agent': user_agent}
class TheFriendlyNeighbourhoodSpider(CrawlSpider):
    name = 'TheFriendlyNeighbourhoodSpider'
    allowed_domains = ['helpnetsecurity.com']
    start_urls = ['https://www.helpnetsecurity.com/2020/12/24/us-cybersecurity-2021-challenges/']
    custom_settings = {'LOG_LEVEL': 'INFO'}
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )
    def parse_item(self, response):#adapt for each website
        link = response.url
        print('lien :', link) 
        article = NewsPlease.from_url(link)
        filename = '../storage/embedding/tout_2.txt'
        with open(filename, 'a') as f:
            m = article.maintext
            f.write('\n\n'+m)