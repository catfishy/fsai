/**
* NBAStats
* @namespace fsai.NBAStats.services
*/
(function () {
  'use strict';

  angular
    .module('fsai.nba.services')
    .factory('NBAStats', NBAStats);

  NBAStats.$inject = ['$cookies', '$http', '$filter'];

  /**
  * @namespace NBAStats
  * @returns {Factory}
  */
  function NBAStats($cookies, $http, $filter) {
    /**
    * @name NBAStats
    * @desc The Factory to be returned
    */
    var NBAStats = {
      getTeamStats: getTeamStats,
      getPlayerStats: getPlayerStats
    };

    return NBAStats;

    /////////////////////

    /**
    * @name getTeamStats
    * @desc Gets team stats for the given date range
    * @param {Object} team stat rows for games in date range
    * @returns {Promise}
    * @memberOf fsai.nba.services.NBAStats
    */
    function getTeamStats(startdate, enddate, canceler) {
      return $http.get('/api/v1/nba/daily-team', {
        params: {
          from: $filter('date')(startdate, "yyyy-MM-dd"),
          to: $filter('date')(enddate, "yyyy-MM-dd"),
          type: 'basic'
        },
        timeout: canceler.promise
      });
    }

    /**
    * @name getTeamMatchupStat
    * @desc Gets team matchup stats for a given game
    * @param {Object} team matchup stat row for given game
    * @returns {Promise}
    * @memberOf fsai.nba.services.NBAStats
    */
    function getTeamStats(arg_tid, arg_gid, canceler) {
      return $http.get('/api/v1/nba/daily-team', {
        params: {
          tid: arg_tid,
          gid: arg_gid,
          type: 'matchup'
        },
        timeout: canceler.promise
      });
    }


    /**
    * @name getPlayerStats
    * @desc Gets team stats for the given date range
    * @param {Object} team stat rows for games in date range
    * @returns {Promise}
    * @memberOf fsai.nba.services.NBAStats
    */
    function getPlayerStats(startdate, enddate, canceler) {
      return $http.get('/api/v1/nba/daily-player', {
        params: {
          from: $filter('date')(startdate, "yyyy-MM-dd"),
          to: $filter('date')(enddate, "yyyy-MM-dd"),
          type: 'basic'
        },
        timeout: canceler.promise
      });
    }

    /**
    * @name getPlayerMatchupStat
    * @desc Gets player matchup stats for a given game
    * @param {Object} player matchup stat row for given game
    * @returns {Promise}
    * @memberOf fsai.nba.services.NBAStats
    */
    function getTeamStats(arg_pid, arg_tid, arg_gid, canceler) {
      return $http.get('/api/v1/nba/daily-player', {
        params: {
          pid: arg_pid,
          tid: arg_tid,
          gid: arg_gid,
          type: 'matchup'
        },
        timeout: canceler.promise
      });
    }

  }
})();
