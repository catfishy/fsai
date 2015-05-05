import sys
import multiprocessing as mp
import logging
import Queue
import re
import math
from datetime import datetime, timedelta
import calendar
import time

from bs4 import BeautifulSoup, element
import requests

class MultiprocessCrawler(object):

    '''
    Multi-processed crawler
    '''

    def __init__(self, name, logger, queue, crawler, num_processes=None):
        self.name = name
        self.logger = logger
        self.queue = queue
        self.result_queue = mp.Queue()
        self.processes = []
        self.crawler = crawler
        self.logger.info("Crawler item limit: %s" % self.crawler.limit)
        self.logger.info("num processes: %s" % num_processes)
        if num_processes:
            self._createProcesses(num_processes)
        if self.crawler.limit:
            num_proc = num_processes or 1
            self.process_limit = math.ceil(self.crawler.limit / float(num_proc))
        else:
            self.process_limit = None

    def queue_listen(self, in_queue, out_queue):
        count = 0
        dropped = 0
        if self.process_limit:
            self.logger.info("PROCESS ITEM LIMIT: %s" % self.process_limit)
        while True:
            try:
                obj = in_queue.get(True, 10)
                if self.crawler.crawlPage(obj):
                    count += 1
                    if self.process_limit and count > self.process_limit:
                        break
                else:
                    dropped += 1
            except Queue.Empty:
                self.logger.info("Queue empty")
                break
            except KeyboardInterrupt:
                sys.exit(1)
            except Exception as e:
                dropped += 1
                self.logger.exception('Error while processing:')
        out_queue.put((count, dropped))

    def _createProcesses(self, num_processes):
        for i in range(num_processes):
            import_process = mp.Process(
                target=self.queue_listen, args=(self.queue, self.result_queue))
            self.processes.append(import_process)

    def start(self):
        for p in self.processes:
            p.start()
            self.logger.info('Started importer process pid:%s', p.pid)

    def join(self):
        for p in self.processes:
            p.join()
            self.logger.info("Joined importer process pid:%s" % p.pid)

    def terminate(self):
        for p in self.processes:
            self.logger.info('Terminating importer process pid:%s', p.pid)
            p.terminate()

    def get_result(self):
        count = 0
        dropped = 0
        while True:
            try:
                new_count, new_dropped = self.result_queue.get(False)
                count += new_count
                dropped += new_dropped
            except Queue.Empty:
                break
        return (count, dropped)

class Crawler(object):

    INIT_PAGE = ''
    LINK_BASE = ''
    BLACKLIST = []
    WHITELIST = []
    PROCESSES = 4

    def __init__(self, logger=None, limit=None):
        self.visited = set()
        self.added = set()
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.url_check_lock = mp.Lock()
        self.url_add_lock = mp.Lock()
        self.name = "basicCrawler"
        self.mp_crawler = None
        self.logger = logger
        self.limit = limit

        self.month_conv = {b:a for a,b in list(enumerate(list(calendar.month_abbr)))[1:]}

    @classmethod
    def get_init_page(cls):
        return cls.INIT_PAGE

    @classmethod
    def get_blacklist(cls):
        return cls.BLACKLIST

    @classmethod
    def get_whitelist(cls):
        return cls.WHITELIST

    def convert_html_table_to_dict(self, table, use_data_stat=False, use_csk=False):
        dict_list = []
        rows = table('tr',recursive=False)
        header_row = rows[0]
        if use_data_stat:
            headers = [col['data-stat'] for col in header_row.contents if type(col) == element.Tag]
        else:
            headers = [col.text for col in header_row.contents if type(col) == element.Tag]
        headers = [h.strip() if h else u'' for h in headers]
        for row in rows[1:]:
            cols = row('td',recursive=False)
            if use_csk:
                values = []
                for col in cols:
                    if type(col) == element.Tag:
                        try:
                            v = col['csk']
                        except KeyError as e:
                            v = None
                        values.append(v)
            else:
                values = [col.text for col in cols if type(col) == element.Tag]
            for i, v in enumerate(values):
                try:
                    values[i] = float(v)
                except Exception as e:
                    values[i] = v
            dict_list.append(dict(zip(headers, values)))
        return dict_list

    def createLogger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('%(asctime)s[%(levelname)s][%(name)s] %(message)s'))
            self.logger.addHandler(ch)

    def extract_links(self, soup):
        '''
        Extract all links in soup
        '''
        links = [link.get('href') for link in soup.findAll('a')]
        return links

    def run(self):
        '''
        run multiprocessed crawling
        '''
        # create logger
        if not self.logger:
            self.createLogger()
        self.logger.info("INIT: %s" % self.INIT_PAGE)
        self.logger.info("BLACKLIST: %s" % self.BLACKLIST)
        self.logger.info("WHITELIST: %s" % self.WHITELIST)
        # add initial page to queue
        self.queue.put(self.INIT_PAGE,block=True)
        self.added.add(self.INIT_PAGE)
        self.mp_crawler = MultiprocessCrawler(type(self).__name__, self.logger, self.queue, self, num_processes=self.PROCESSES)
        self.mp_crawler.start()
        self.mp_crawler.join()

    def checkVisit(self, url):
        '''
        Check if the url has been visited,
        if not, grab content and convert to soup
        '''
        self.url_check_lock.acquire()
        # check if in visited
        if url in self.visited:
            self.logger.info("%s already visited" % url)
            self.url_check_lock.release()
            return False
        # get content
        try:
            init_soup = self.getContent(url)
            self.visited.add(url)
        except Exception as e:
            self.logger.info("%s no content: %s" % (url,e))
            self.url_check_lock.release()
            return False
        self.url_check_lock.release()
        return init_soup

    def getContent(self, url):
        init_response = requests.get(url, timeout=10)
        init_content = init_response.content
        init_soup = BeautifulSoup(init_content)
        return init_soup

    def checkAdd(self, url):
        self.url_add_lock.acquire()
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
        urls_added = []
        for raw in links:
            if not raw:
                continue
            # pass blacklist
            black_matches = [black in raw for black in self.BLACKLIST]
            if any(black_matches):
                continue
            white_matches = [white in raw for white in self.WHITELIST]
            if not any(white_matches):
                continue
            # construct full link
            full = self.LINK_BASE + raw

            # check visit
            if full in self.added or full in self.visited:
                continue

            # try to add to queue
            added = self.checkAdd(full)
            
            if added:
                urls_added.append(full)
                links_added += 1
        #self.logger.info("Added to queue: %s" % urls_added)
        return links_added

    def crawlPage(self, url):
        raise NotImplementedError()

