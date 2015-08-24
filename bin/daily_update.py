from optparse import OptionParser
import sys
import logging
from datetime import datetime

from statsETL.bball.BRcrawler import playerCrawler, teamCrawler, gameCrawler
from statsETL.bball.NBAcrawler import crawlUpcomingGames, crawlNBATrackingStats, saveNBADepthChart
from analysis.util.kimono import getESPNTeamStats, updateNBARosters
#from analysis.bball.gameAnalysis import modelPlayersInUpcomingGames


if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-m', '--model', dest="model", action='store_true', default=False,
                      help="Run model projections")
    parser.add_option('-c', '--crawl', dest="crawl", action='store_true', default=False,
                      help="Run all crawlers")
    parser.add_option('-e', '--essential', dest="essential", action='store_true', default=False,
                      help="Run essential crawlers")
    parser.add_option('-s', '--small', dest="small", action='store_true', default=False,
                      help="Do small hourly run")
    parser.add_option("-p", "--poolsize", dest="poolsize", default=2,
                      help="Modeling pool size")

    (options, args) = parser.parse_args()

    model = options.model
    crawl = options.crawl
    poolsize = int(options.poolsize)
    essential = options.essential
    small = options.small

    if small:
        g_crawl = gameCrawler(refresh=True, days_back=3, end_date=datetime.now())
        g_crawl.run()
        crawlUpcomingGames(days_ahead=7)
        updateNBARosters()
        getESPNTeamStats()
        crawlNBATrackingStats()
        saveNBADepthChart()
    elif essential:
        g_crawl = gameCrawler(refresh=True, days_back=1100, end_date=datetime(year=2013,month=3,day=20))
        g_crawl.run()
        crawlUpcomingGames(days_ahead=7)
        updateNBARosters()
        getESPNTeamStats()
        crawlNBATrackingStats()
        saveNBADepthChart()
    elif crawl:
        p_crawl = playerCrawler(refresh=True)
        t_crawl = teamCrawler(refresh=True)
        g_crawl = gameCrawler(refresh=True, days_back=2000, end_date=datetime(year=2015,month=4,day=18))
        #g_crawl.run()
        #p_crawl.run()
        p_crawl.run()
    if model:
        # create logger
        logger = logging.getLogger("player_modeling")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter('%(asctime)s[%(levelname)s][%(name)s] %(message)s'))
            logger.addHandler(ch)

        modelPlayersInUpcomingGames(logger, days_ahead=2, poolsize=poolsize)