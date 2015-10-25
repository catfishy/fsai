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
      getTeamStats: getTeamStats
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
    function getTeamStats(startdate, enddate) {
      return $http.get('/api/v1/nba/daily-team-own', {
        params: {
          from: $filter('date')(startdate, "yyyy-MM-dd"),
          to: $filter('date')(enddate, "yyyy-MM-dd")
        }
      });
    }
  }
})();
