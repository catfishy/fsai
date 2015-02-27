from optparse import OptionParser

from statsETL.bball.BRcrawler import playerCrawler, teamCrawler, gameCrawler
from statsETL.bball.NBAcrawler import crawlUpcomingGames
from analysis.util.kimono import updateNBARosters
from analysis.bball.gameAnalysis import modelPlayersInUpcomingGames


if __name__=="__main__":
    parser = OptionParser()
    parser.add_option(
        '-m', '--model', dest="model", action='store_true', default=False,
        help="Run model projections")
    parser.add_option(
        '-c', '--crawl', dest="crawl", action='store_true', default=False,
        help="Run crawlers")

    (options, args) = parser.parse_args()

    model = options.model
    crawl = options.crawl

    if crawl:
        p_crawl = playerCrawler(refresh=True)
        t_crawl = teamCrawler(refresh=True)
        g_crawl = gameCrawler(refresh=True, days_back=7)
        p_crawl.run()
        g_crawl.run()
        t_crawl.run()
        updateNBARosters()
        crawlUpcomingGames(days_ahead=7)
    if model:
        modelPlayersInUpcomingGames(days_ahead=3)