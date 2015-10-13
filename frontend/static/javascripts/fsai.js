(function () {
  'use strict';

    angular
        .module('fsai', [
          'fsai.routes',
          'fsai.authentication',
          'fsai.config'
        ]);

    angular
        .module('fsai.routes', ['ngRoute']);

    angular
        .module('fsai.config', []);


})();