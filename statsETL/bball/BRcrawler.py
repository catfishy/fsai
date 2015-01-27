import sys
import multiprocessing as mp
import logging
import Queue

from bs4 import BeautifulSoup
import requests


class Crawler:

    INIT_PAGE = ''
    LINK_BASE = ''
    BLACKLIST = []
    WHITELIST = []
    PROCESSES = 4

    def __init__(self):
        self.visited = set()
        self.added = set()
        self.manager = mp.Manager()
        self.queue = self.manager.Queue(maxsize=1000)
        self.url_check_lock = mp.Lock()
        self.url_add_lock = mp.Lock()

    @classmethod
    def get_init_page(cls):
        return cls.INIT_PAGE

    @classmethod
    def get_blacklist(cls):
        return cls.BLACKLIST

    @classmethod
    def get_whitelist(cls):
        return cls.WHITELIST

    def extract_links(self, soup):
        '''
        Extract all links in soup
        '''
        for link in soup.findAll('a'):
            print link.get('href')

    def run(self):
        '''
        TODO: make multiprocessing work
        '''
        # add initial page to queue
        self.queue.put(self.INIT_PAGE,block=True)
        process_list = []
        for p_num in range(self.processes):
            p = mp.Process(target=self.startCrawling)
            p.start()
            process_list.append(p)
        for proc in process_list:
            proc.join()


    def checkVisit(self, url):
        '''
        Check if the url has been visited,
        if not, grab content and convert to soup
        '''
        self.url_check_lock.acquire()
        # check if in visited
        if url in self.visited:
            self.url_check_lock.release()
            return False
        # get content
        init_response = requests.get(url)
        init_content = init_response.content
        init_soup = BeautifulSoup(init_content)
        self.visited.add(url)
        self.url_check_lock.release()
        return soup

    def checkAdd(self, url):
        self.url_add_lock.acquire()
        if url in self.added:
            self.url_add_lock.release()
            return False
        self.queue.put(url,block=True)
        self.added.add(url)
        self.url_add_lock.release()
        return True

    def addLinksToQueue(self, links):
        '''
        Construct full links, pass blacklist/whitelist,
        check existence in queue + visited
        '''
        links_added = 0
        for raw in links:
            # construct full link
            full = self.LINK_BASE + raw
            full = full.replace('//','/')
            # pass blacklist
            black_matches = [black in full for black in self.BLACKLIST]
            if any(black_matches):
                continue
            white_matches = [white in full for white in self.WHITELIST]
            if not any(white_matches):
                continue
            added = self.checkAdd(full)
            if added:
                links_added += 1
        return links_added

    def startCrawling(self):
        '''
        pull page from queue and crawl it
        If no page in queue for 10 seconds, then quit

        TODO: run this as multiprocessed function
        '''
        try:
            while True:
                next_url = self.queue.get(block=True, timeout=15)
                self.crawlPage(next_url)
        except Queue.Empty as e:
            logging.log("Queue empty")
            return

    def crawlPage(self, url):
        raise NotImplementedError()


class playerCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/players/"
    LINKBASE = "http://www.basketball-reference.com"
    BLACKLIST = []
    WHITELIST = ["/players/"]
    PROCESSES = 2

    def isPlayerPage(self, soup):
        return False

    def crawlPlayerPage(self, soup):
        pass

    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        soup = self.checkVisit(url)
        if not soup:
            return
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        # decide relevance to crawl
        if self.isPlayerPage(soup):
            self.crawlPlayerPage(soup)


if __name__=="__main__":
    p_crawl = playerCrawler()
    p_crawl.run()


