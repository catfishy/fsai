!function(){"use strict";function a(a){a.defaults.xsrfHeaderName="X-CSRFToken",a.defaults.xsrfCookieName="csrftoken"}angular.module("fsai",["fsai.routes","fsai.authentication","fsai.config","fsai.layout","fsai.posts","fsai.utils","fsai.profiles","fsai.nba"]),angular.module("fsai.routes",["ngRoute"]),angular.module("fsai.config",[]),angular.module("fsai").run(a),a.$inject=["$http"]}();