from optparse import OptionParser
import sys
import logging

from statsETL.bball.BRcrawler import playerCrawler, teamCrawler, gameCrawler
from statsETL.bball.NBAcrawler import crawlUpcomingGames
from analysis.util.kimono import updateNBARosters
from analysis.bball.gameAnalysis import modelPlayersInUpcomingGames


if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-m', '--model', dest="model", action='store_true', default=False,
                      help="Run model projections")
    parser.add_option('-c', '--crawl', dest="crawl", action='store_true', default=False,
                      help="Run all crawlers")
    parser.add_option('-e', '--essential', dest="essential", action='store_true', default=False,
                      help="Run essential crawlers")
    parser.add_option("-p", "--poolsize", dest="poolsize", default=2,
                      help="Modeling pool size")

    (options, args) = parser.parse_args()

    model = options.model
    crawl = options.crawl
    poolsize = int(options.poolsize)
    essential = options.essential

    if essential:
        g_crawl = gameCrawler(refresh=True, days_back=7)
        g_crawl.run()
        updateNBARosters()
        crawlUpcomingGames(days_ahead=7)
    elif crawl:
        p_crawl = playerCrawler(refresh=True)
        t_crawl = teamCrawler(refresh=True)
        g_crawl = gameCrawler(refresh=True, days_back=7)
        #p_crawl.run()
        g_crawl.run()
        #t_crawl.run()
        #updateNBARosters()
        #crawlUpcomingGames(days_ahead=7)
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