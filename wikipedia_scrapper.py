from bs4 import BeautifulSoup
import urllib2
import re


class wiki_scrapper(object):
    
    """scrapper object: takes in a list of any number of
    wikipedia categories to scrap, and the maximum level 
    of recursion to go through subcategories"""
    
    def __init__(self,categories,max_lvls):
        
        base_url = "https://en.wikipedia.org/wiki/Category:"
        self.urls = [base_url+cat for cat in categories]
        self.max_lvls = max_lvls
        self.viewed_categories = set(self.urls)
        self.banned_words = ['Main_Page']
        self.banned_categories = ['/wiki/Category:Commons_category_with_local_link_same_as_on_Wikidata',
                           '/wiki/Category:Commons_category_with_local_link_different_than_on_Wikidata']
        self.data = list()
        self.target = list()
        self.level = 0
        
    
    def fetch_url_text(self,url):
        """
        scrap text for a single url
        """
        
        html_cat = urllib2.urlopen(url)
        soup_cat = BeautifulSoup(html_cat,"html5lib")
        
        #go over the page links in current page an scrap text
        for link in soup_cat.findAll('a', attrs={'href': re.compile("^/wiki/")}):
            if not any(word in self.banned_words for word in link.get('href').split('/')) and not ':' in link.get('href'):

                html_page = urllib2.urlopen("https://en.wikipedia.org"+link.get('href'))
                soup_page = BeautifulSoup(html_page,"html5lib")
                word_list = list()
                raw_text = soup_page.find_all('p')

                for lines in  raw_text:   
                    word_list.append(lines.text)

                return " ".join(word_list)
    
    def fetch_cat_text(self,url,nclass,level):
        
        """Recursively go through category, 
            subcategoires, and next pages to scrap text"""
        
        level+=1 #increment subcategory level at each recursion
        html_cat = urllib2.urlopen(url)
        soup_cat = BeautifulSoup(html_cat,"html5lib")
        
        #go over the next category page if exists
        nextpage = soup_cat.findAll('a',attrs={'href': re.compile("pagefrom")}) 
        if nextpage:
            url = "https://en.wikipedia.org"+nextpage[0].get('href')
            if url not in self.viewed_categories:
                self.viewed_categories.add(url)
                self.fetch_cat_text(url,nclass,level)
           
        #go over the subcategories present in the current page, ignore those at the bottom
        subcategories = soup_cat.findAll('a', attrs={'href': re.compile("^/wiki/Category")})
        subcategories = [s for n,s in enumerate(subcategories) if s not in subcategories[:n]]
       
        for subcat in subcategories[:-6]:
            #check that the subcategory level is not too deep
            if level <= self.max_lvls[nclass-1] and not any(banned_category in subcat.get('href')
                                            for banned_category in self.banned_categories):
                
                url_sub = "https://en.wikipedia.org"+subcat.get('href')

                if url_sub not in self.viewed_categories: 
                    print "             subcategory "+url_sub.split(':')[2]
                    self.viewed_categories.add(url_sub)
                    self.fetch_cat_text(url_sub,nclass,level)

        #go over the page links in current page an scrap text
        for link in soup_cat.findAll('a', attrs={'href': re.compile("^/wiki/")}):
            if not any(word in self.banned_words for word in link.get('href').split('/')) and not ':' in link.get('href'):

                html_page = urllib2.urlopen("https://en.wikipedia.org"+link.get('href'))
                soup_page = BeautifulSoup(html_page,"html5lib")
                word_list = list()
                raw_text = soup_page.find_all('p')
                for lines in  raw_text:   
                    word_list.append(lines.text)

                self.data.append(' '.join(word_list))
                self.target.append(nclass)
                   
                    
    def get_data(self):
        
        """get text data for each category"""
        
        for nclass,url in enumerate(self.urls):
            print "scrapping category: "+url.split(':')[2]
            self.fetch_cat_text(url,nclass+1,level=0)

        return self.data, self.target
