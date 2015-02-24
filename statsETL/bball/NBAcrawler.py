from datetime import datetime, timedelta

from statsETL.bball.BRcrawler import Crawler
from statsETL.db.mongolib import *

class upcomingGameCrawler(Crawler):

    INIT_PAGE = "http://www.nba.com/gameline/"

    def __init__(self, date=None, logger=None):
        super(upcomingGameCrawler, self).__init__(logger=logger)
        self.name = "gamelineCrawler"
        if not date:
            raise Exception("Must Specify a datetime")
        self.date = date # must be datetime object


    def crawlPage(self):
        if not self.logger:
            self.createLogger()
        # create url to crawl
        year = str(self.date.year)
        month = str(self.date.month).zfill(2)
        day = str(self.date.day).zfill(2)
        url_to_crawl = "%s%s%s%s" % (self.INIT_PAGE, year, month, day)
        print url_to_crawl

        # try to get the soup
        soup = self.checkVisit(url_to_crawl)

        if not soup:
            self.logger.info("could not find url %s" % url_to_crawl)
            return False

        return self.crawlGamelinePage(url_to_crawl, soup)

    def crawlGamelinePage(self, url, soup):
        recap_listings = soup('div', class_='Recap GameLine')
        live_listings = soup('div', class_='Live GameLine')
        preview_listings = soup('div', class_='Pre GameLine')
        
        preview_data = []

        # for preview boxes
        for preview in preview_listings:
            prescore_div = preview.find('div', class_="nbaPreMnScore")
            time_div = prescore_div.find('div', class_="nbaPreMnStatus")
            teams_div = prescore_div.find('div', class_="nbaPreMnTeamInfo")

            time_parts = [x for x in time_div.stripped_strings]
            teams_images = teams_div('img', title=True)
            teams_parts = [x['title'] for x in teams_images]

            gametime = self.parseTime(time_parts)
            data = {'time': gametime,
                    'home_team_name': teams_parts[1],
                    'away_team_name': teams_parts[0]
                    }
            preview_data.append(data)

        # for recap boxes
        for recap in recap_listings:
            score_div = recap.find('div', class_="nbaModTopScore")
            time_div = score_div.find('div', class_="nbaFnlStatTxSm")
            away_team_div = score_div.find('div', class_="nbaModTopTeamAw")
            home_team_div = score_div.find('div', class_="nbaModTopTeamHm")
            away_image = away_team_div.find('img', title=True)
            home_image = home_team_div.find('img', title=True)
            away_name = away_image['title']
            home_name = home_image['title']

            time_parts = [x for x in time_div.stripped_strings]
            time_parts = time_parts[0].split(' ')
            gametime = self.parseTime(time_parts)
            data = {'time': gametime,
                    'home_team_name': home_name,
                    'away_team_name': away_name
                    }
            preview_data.append(data)

        # for live games
        for live in live_listings:
            pass

        # save into future games db
        for d in preview_data:
            home_row = team_collection.find_one({"name":str(d['home_team_name']).strip()})
            away_row = team_collection.find_one({"name":str(d['away_team_name']).strip()})
            home_id = home_row['_id']
            away_id = away_row['_id']
            teams = [home_id, away_id]
            new_id = "%s@%s" % (away_id, home_id)
            d['_id'] = new_id
            d['teams'] = teams
            d['home_id'] = home_id
            d['away_id'] = away_id
            nba_conn.saveDocument(future_collection, d)

        return preview_data

    def parseTime(self, time_parts):
        hourminute = time_parts[0].split(':')
        ampm = time_parts[1]
        hour = int(hourminute[0])
        if 'pm' in ampm:
            hour += 12
        minute = int(hourminute[1])
        gametime = datetime(year=self.date.year, month=self.date.month, day=self.date.day, hour=hour, minute=minute)
        return gametime

if __name__=="__main__":
    today = datetime.now() + timedelta(int(1))
    gl_crawl = upcomingGameCrawler(date=today)
    gl_crawl.crawlPage()


