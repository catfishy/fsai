'''
Crawl for new games + upcoming games
'''
from datetime import datetime, timedelta

from statsETL.bball.NBAcrawler import crawlUpcomingGames, crawlNBAGames

DAYS_AHEAD = 10
DAYS_BEHIND = 10

if __name__ == "__main__":
    # crawl ahead
    upcoming = crawlUpcomingGames(days_ahead=DAYS_AHEAD)
    print "UPCOMING CRAWLED: %s" % upcoming

    # crawl behind
    today = datetime.now()
    start = today - timedelta(days=DAYS_BEHIND)
    crawlNBAGames(start.strftime("%m/%d/%Y"),today.strftime("%m/%d/%Y"))
