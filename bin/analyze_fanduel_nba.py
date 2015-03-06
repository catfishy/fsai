from optparse import OptionParser
import sys
import logging

from analysis.bball.gameAnalysis import analyzeFanDuelGame, graphUpcomingGames


if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-g", "--gameid", dest="gameid", default=None,
                      help="Fanduel game id")
    parser.add_option("-c", "--crawl", dest="crawl", default=False,
                      action='store_true')

    (options, args) = parser.parse_args()

    gameid = options.gameid
    crawl = options.crawl

    if not gameid:
        raise Exception("Specify a Fanduel NBA matchup ID")

    # create logger
    logger = logging.getLogger("fanduel_nba")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(
            logging.Formatter('%(asctime)s[%(levelname)s][%(name)s] %(message)s'))
        logger.addHandler(ch)

    #analyzeFanDuelGame(gameid, logger, crawl=crawl)
    graphUpcomingGames(gameid)

