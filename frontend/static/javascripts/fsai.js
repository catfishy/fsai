(function () {
  'use strict';

    angular
        .module('fsai', [
          'smart-table',
          'fsai.routes',
          'fsai.authentication',
          'fsai.config',
          'fsai.layout',
          'fsai.posts',
          'fsai.utils',
          'fsai.profiles',
          'fsai.nba'
        ]);

    angular
        .module('fsai.routes', ['ngRoute']);

    angular
        .module('fsai.config', []);

    angular
        .module('fsai')
        .run(run);



    run.$inject = ['$http'];

    /**
    * @name run
    * @desc Update xsrf $http headers to align with Django's defaults
    */
    function run($http) {
      $http.defaults.xsrfHeaderName = 'X-CSRFToken';
      $http.defaults.xsrfCookieName = 'csrftoken';
    }


})();