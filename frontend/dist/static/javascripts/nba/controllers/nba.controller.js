!function(){"use strict";function t(t,e,a){function n(){d.startdate=new Date,d.enddate=new Date((new Date).getTime()+864e5)}function s(){data=a.getTeamStats(d.startdate,d.enddate),d.stat_headers=[{name:"GAMEID",key:"gid"},{name:"TEAMID",key:"tid"},{name:"GAMEID",key:"gid"},{name:"TEAMID",key:"tid"},{name:"GAMEID",key:"gid"},{name:"TEAMID",key:"tid"},{name:"GAMEID",key:"gid"},{name:"TEAMID",key:"tid"}],d.stat_rows=[],$.each(data,function(t,e){$.each(e,function(e,a){a.gid=t,a.tid=e,d.stat_rows.push(a)}),console.log(d.stat_headers),console.log(d.stat_rows)})}var d=this;d.startdate=void 0,d.enddate=void 0,d.populateTeamStats=s,d.itemsByPage=20,d.stat_headers=[],d.stat_rows=[],d.displayed_headers=[].concat(d.stat_headers),d.displayed_rows=[].concat(d.stat_rows),n()}angular.module("fsai.nba.controllers").controller("NBAController",t),t.$inject=["$location","Snackbar","NBAStats"]}();