!function(){"use strict";function n(n,t,o){function i(){o.isAuthenticated()&&n.url("/")}function c(){o.login(l.email,l.password)}var l=this;l.login=c,i()}angular.module("fsai.authentication.controllers").controller("LoginController",n),n.$inject=["$location","$scope","Authentication"]}();