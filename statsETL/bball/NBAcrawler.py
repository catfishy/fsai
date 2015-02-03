from datetime import datetime, timedelta

from statsETL.bball.BRcrawler import Crawler


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
        
        # for preview boxes
        preview_data = []
        for preview in preview_listings:
            prescore_div = preview.find('div', class_="nbaPreMnScore")
            time_div = prescore_div.find('div', class_="nbaPreMnStatus")
            teams_div = prescore_div.find('div', class_="nbaPreMnTeamInfo")

            time_parts = [x for x in time_div.stripped_strings]
            teams_parts = [x for x in teams_div.stripped_strings]

            hourminute = time_parts[0].split(':')
            ampm = time_parts[1]
            hour = int(hourminute[0])
            if 'pm' in ampm:
                hour += 12
            minute = int(hourminute[1])
            gametime = datetime(year=self.date.year, month=self.date.month, day=self.date.day, hour=hour, minute=minute)

            data = {'time': gametime,
                    'home_team_id': teams_parts[1],
                    'away_team_id': teams_parts[0]
                    }
            preview_data.append(data)
        return preview_data


if __name__=="__main__":
    today = datetime.now() + timedelta(int(1))
    gl_crawl = upcomingGameCrawler(date=today)
    gl_crawl.crawlPage()


