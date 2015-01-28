import sys
import multiprocessing as mp
import logging
import Queue
import re
import math
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

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

    @classmethod
    def get_init_page(cls):
        return cls.INIT_PAGE

    @classmethod
    def get_blacklist(cls):
        return cls.BLACKLIST

    @classmethod
    def get_whitelist(cls):
        return cls.WHITELIST

    def convert_html_table_to_dict(self, table):
        dict_list = []
        rows = table('tr',recursive=False)
        header_row = rows[0]
        headers = [col.string for col in header_row.contents if type(col) == element.Tag]
        headers = [h.strip() if h else u'' for h in headers]
        for row in rows[1:]:
            cols = row('td',recursive=False)
            values = [col.string for col in cols if type(col) == element.Tag]
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
            init_response = requests.get(url, timeout=10)
            init_content = init_response.content
            init_soup = BeautifulSoup(init_content)
            self.visited.add(url)
        except Exception as e:
            self.logger.info("%s no content: %s" % (url,e))
            self.url_check_lock.release()
            return False
        self.url_check_lock.release()
        return init_soup

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
            added = self.checkAdd(full)
            if added:
                #self.logger.info("newlink: %s" % full)
                links_added += 1
        return links_added

    def crawlPage(self, url):
        raise NotImplementedError()



class teamCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/teams/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ['html']
    WHITELIST = ["/teams/"]
    PROCESSES = 2

    def __init__(self, logger=None):
        super(teamCrawler, self).__init__(logger=logger)
        self.name = "teamCrawler"

    def isTeamPage(self, url, soup):
        '''
        - check url fits pattern
        '''
        # check url
        url_pattern = re.compile("/teams/[A-Z]+/$")
        result = url_pattern.search(url)
        if not result:
            return False
        return True

    def crawlTeamPage(self, url, soup):
        team_id = url.split('/')[-2].strip()
        stw_boxes = soup("div", class_="stw")
        stw_box = stw_boxes[0]
        # get first p
        p_s = stw_box("p")[0]
        loc_span = p_s("span")[0]
        loc_string = loc_span.next_sibling.strip()
        name_span = p_s("span")[1]
        name_strings = name_span.next_sibling.split(',')
        recent_name = name_strings[0].strip()
        data = {'location': loc_string,
                'name': recent_name,
                'team_id': team_id,
                'url': url}
        self.logger.info("SAVING: %s" % data)
        return data

    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        soup = self.checkVisit(url)
        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isTeamPage(url, soup):
            self.crawlTeamPage(url, soup)
            return True
        return False

class gameCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/boxscores/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ['pbp','shot-chart','plus-minus']
    WHITELIST = ["/boxscores/"]
    PROCESSES = 2

    def __init__(self, logger=None, limit=None):
        super(gameCrawler, self).__init__(logger=logger, limit=limit)
        self.name = "gameCrawler"
        # add tomorrow to blacklist to prevent going into the future
        # each box score page only has links to day before and after
        now = datetime.now() + timedelta(1) # one day from now
        day = now.day
        month = now.month
        year = now.year
        dateblacklist = "index.cgi?month=%s&day=%s&year=%s" % (day,month,year)
        self.BLACKLIST.append(dateblacklist)

    def isGamePage(self, url, soup):
        '''
        - check url fits pattern
        '''
        # check url
        url_pattern = re.compile("/boxscores/[0-9]+[A-Z]+\.html$")
        result = url_pattern.search(url)
        if not result:
            return False
        return True

    def crawlGamePage(self, url, soup):
        game_id = url.split('/')[-1].replace('.html','').strip().upper()
        page_content = soup.find('div', id="page_content").find('table')
        all_tables = page_content.find('tr').find('td')
        all_divs = all_tables('div',recursive=False)
        boxscore_div = all_divs[0]
        stats_div = all_divs[1]

        # parse boxscore
        float_left = boxscore_div.find('div', class_='float_left')
        box_tables = float_left('table', recursive=False)
        final_table = box_tables[1].find('tr').find('td').find('table')
        final_insides = final_table('tr', recursive=False)
        finalscore_table = final_insides[0]
        finalscore_table_insides = finalscore_table('td', recursive=False)
        team1_finalscore = finalscore_table_insides[0]('span')[0].contents
        team2_finalscore = finalscore_table_insides[1]('span')[0].contents
        gameinfo_table = final_insides[1].find('td').contents
        gametime = gameinfo_table[0]
        gamelocation = gameinfo_table[1].string
        team_id_regex = re.compile("/[A-Z]+/")
        team1_id = team_id_regex.search(team1_finalscore[0]['href']).group(0).replace('/','')
        team2_id = team_id_regex.search(team2_finalscore[0]['href']).group(0).replace('/','')
        team1_pts = team1_finalscore[1].string
        team2_pts = team2_finalscore[1].string

        data = {'game_id': game_id,
                'team1_id': team1_id,
                'team2_id': team2_id,
                'team1_pts': team1_pts,
                'team2_pts': team2_pts,
                'time': gametime,
                'location': gamelocation,
                'url': url}

        teamstat_table = box_tables[2].find('tr')
        teamstat_insides = teamstat_table('td', recursive=False)
        scoring_table = teamstat_insides[0].find('table')
        factors_table = teamstat_insides[1].find('table')
        # remove first row from scoring table
        scoring_table('tr')[0].extract()
        # remove colgroup, unwrap thead and tbody, then remove first row
        factors_table('colgroup')[0].extract()
        factors_table.thead.unwrap()
        factors_table.tbody.unwrap()
        factors_table('tr')[0].extract()

        scoring_dict = self.convert_html_table_to_dict(scoring_table)
        factors_dict = self.convert_html_table_to_dict(factors_table)
        
        # empty string key is the team id
        for d in scoring_dict:
            d['team_id'] = d['']
            d.pop('')
        for d in factors_dict:
            d['team_id'] = d['']
            d.pop('')

        # TODO: save scoring_dict and factors_dict to respective team/game stat rows

        # parse player stats
        stats_tables = stats_div('div', class_='table_container')
        stats_tables_by_id = {tablediv['id'].replace('div_','') : tablediv.table for tablediv in stats_tables}
        for k,v in stats_tables_by_id.iteritems():
            # remove colgroup
            v('colgroup')[0].extract()
            v.thead.unwrap()
            v.tbody.unwrap()
            v('tr')[0].extract()
            v_dict = [_ for _ in self.convert_html_table_to_dict(v) if _]
            # replace u'Starters' key with "player_name"
            for d in v_dict:
                d['player_name'] = d['Starters']
                d.pop('Starters')
            stats_tables_by_id[k] = v_dict

        # TODO: save stats_tables_by_id to respective player/game stat rows

        # parse game info at bottom of page
        lower_gameinfo_table = stats_div('table', recursive=False)[0]
        for row in lower_gameinfo_table('tr', recursive=False):
            title_name = row('td')[0].string.lower().replace(':','')
            if title_name == 'attendance':
                data['attendance'] = row('td')[1].string
            elif title_name == 'time of game':
                data['time of game'] = row('td')[1].string
            elif title_name == 'officials':
                official_links = row('td')[1]('a', recursive=False)
                official_ids = [l['href'].split('/')[-1].replace('.html','') for l in official_links]
                data['officials'] = official_ids
            elif title_name == 'inactive':
                inactive_links = row('td')[1]('a', recursive=False)
                inactive_ids = [l['href'].split('/')[-1].replace('.html','') for l in inactive_links]
                data['inactive'] = inactive_ids

        self.logger.info("SAVING: %s" % data)
        return data


    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        soup = self.checkVisit(url)
        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isGamePage(url, soup):
            self.crawlGamePage(url, soup)
            return True
        return False


class playerCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/players/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ["gamelog", "splits", "news.cgi","shooting", "lineups", "on-off", "cbb", "http", "nbdl", "euro", "nbl"]
    WHITELIST = ["/players/"]
    PROCESSES = 2

    def __init__(self, logger=None):
        super(playerCrawler, self).__init__(logger=logger)
        self.name = "playerCrawler"

    def isPlayerPage(self, url, soup):
        '''
        - check url fits pattern
        - look for experience in info_box (for active player)
        '''
        # check url
        url_pattern = re.compile("/players/[a-z]/[a-z]+[0-9]+\.html")
        result = url_pattern.search(url)
        if not result:
            return False
        # check info_box
        info_boxes = soup(id="info_box")
        if len(info_boxes) > 0:
            info_box = info_boxes[0]
            spans = info_box(text=re.compile("Experience\:"))
            if len(spans) > 0:
                return True
        return False

    def crawlPlayerPage(self, url, soup):
        player_id = url.split('/')[-1].replace('.html','')
        self.logger.info("Crawling PLAYER: %s" % player_id)
        info_boxes = soup(id="info_box")
        info_box = info_boxes[0]
        name_p = info_box("p", class_="margin_top")[0]
        full_name = name_p("span", class_="bold_text")[0].string
        stat_p = info_box("p", class_="padding_bottom_half")[0]
        stat_titles = stat_p("span", class_="bold_text")
        data = {'player_id': player_id,
                'full_name': full_name,
                'url': url
                }
        valid_fields = ['position','shoots','height','weight','born','nba_debut','experience']
        for title_tag in stat_titles:
            title = title_tag.string
            title = title.replace(':','').lower().strip()
            if title in valid_fields:
                if title == 'born':
                    birth_span = soup(id="necro-birth")[0]
                    text = birth_span['data-birth']
                elif title == 'nba_debut':
                    date_link = title_tag.next_sibling
                    date = date_link.string
                    text = date
                elif title == 'experience':
                    text = title_tag.next_sibling.strip().lower()
                    if text == 'rookie':
                        text = '0'
                    else:
                        text = text.replace('years','').strip()
                elif title == 'weight':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace('lbs.','').strip()
                elif title == 'height':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace(u"\xa0\u25aa", u' ').strip()
                elif title == 'position':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace(u"\xa0\u25aa", u' ').strip()
                else:
                    text = title_tag.next_sibling.strip()
                data[title] = text
        self.logger.info("SAVING: %s" % data)
        return data


    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        soup = self.checkVisit(url)
        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isPlayerPage(url, soup):
            self.crawlPlayerPage(url, soup)
            return True
        return False


if __name__=="__main__":
    p_crawl = playerCrawler()
    g_crawl = gameCrawler(limit=10)
    t_crawl = teamCrawler()

    #p_crawl.run()
    g_crawl.run()
    #t_crawl.run()


