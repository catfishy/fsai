'''
Crawl for new games + upcoming games
'''
from datetime import datetime, timedelta

from statsETL.bball.NBAcrawler import crawlUpcomingGames, crawlNBAGames, crawlNBAPlayerData, crawlNBARoster

DAYS_AHEAD = 10
DAYS_BEHIND = 20
YEARS = [2015]
RECRAWL= True

if __name__ == "__main__":

    # crawl ahead
    upcoming = crawlUpcomingGames(days_ahead=DAYS_AHEAD)
    print "UPCOMING CRAWLED: %s" % upcoming

    # crawl behind
    today = datetime.now() + timedelta(days=1)
    start = today - timedelta(days=DAYS_BEHIND)
    crawlNBAGames(start.strftime("%m/%d/%Y"),today.strftime("%m/%d/%Y"), recrawl=RECRAWL)

    # crawl player data (shot charts)
    crawlNBAPlayerData(years=YEARS, last_n_days=DAYS_BEHIND)

    # crawl rosters
    crawlNBARoster()
