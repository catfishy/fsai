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
    function getTeamStats(startdate, enddate, types, canceler) {
      return $http.get('/api/v1/nba/daily-team', {
        params: {
          from: $filter('date')(startdate, "yyyy-MM-dd"),
          to: $filter('date')(enddate, "yyyy-MM-dd"),
          types: types.join()
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
    function getPlayerStats(startdate, enddate, types, canceler) {
      return $http.get('/api/v1/nba/daily-player', {
        params: {
          from: $filter('date')(startdate, "yyyy-MM-dd"),
          to: $filter('date')(enddate, "yyyy-MM-dd"),
          types: types.join()
        },
        timeout: canceler.promise
      });
    }



  }
})();
