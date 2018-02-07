
function init_map (config){
    /**
     * Create the map
     */
    var map = AmCharts.makeChart( "chartdiv", config );
    return map;

}


var mouseManager  = new function MousePosition(){


    var x = 0, y = 0;

    var callbacks = [];

    document.querySelector("body").onmousemove = function(mouseEvent){

        x = mouseEvent ? mouseEvent.screenX : typeof(window.event) != 'undefined' ? window.event.screenX : this.x;
        y = mouseEvent ? mouseEvent.screenY : typeof(window.event) != 'undefined' ? window.event.screenY : this.y;

        for(var i = 0; i<callbacks.length; i++){
            callbacks[i](x,y);
        }

    };


    this.addCallback = function(callback){
        callbacks.push(callback);
    };


    this.getX = function(){
        return x;
    };

    this.getY = function(){
        return y;
    };

};


mouseManager.addCallback(function(currX, currY){

    /*
    var isControlerOpen = document.querySelector("aside").style.visibility != "hidden";

    if(currX < 80 && !isControlerOpen){
        document.querySelector("aside").style.visibility = "visible";
    }
    else if(currX > 300 && isControlerOpen){
        document.querySelector("aside").style.visibility = "hidden";
    }
    */


});






var sideMenuManager = new function(){

    var companyResult = null;

    document.addEventListener('DOMContentLoaded', function(){
        companyResult = document.querySelectorAll(".airlines tr");
    }.bind(this));

    this.updateResult = function(sortedResultList){

        if(companyResult == null || sortedResultList.length == 0){
            return;
        }


        var reducer = function(acc, newElem){
            if(isNaN(newElem[1])){
                return acc;
            }
            else {
                return acc + newElem[1];
            }
        };

        var totalCrashes = sortedResultList.reduce(reducer,0.00001);

        for(var i = 0; i< Math.min(sortedResultList.length, companyResult.length); i++){
            // first row dedicated to label
            var cells = companyResult[i].querySelectorAll("td");
            cells[0].innerHTML = sortedResultList[i][0].substr(0,30);
            cells[1].innerHTML = sortedResultList[i][1];
            cells[2].innerHTML = Math.round(parseFloat(sortedResultList[i][1])/totalCrashes*100)+"%";
        }

        for(var i = Math.min(sortedResultList.length, companyResult.length); i<companyResult.length; i++){
            var cells = companyResult[i].querySelectorAll("td");
            cells[0].innerHTML = "";
            cells[1].innerHTML = "";
            cells[2].innerHTML = "";
        }

    };

};




var CrashChart = new function(){

    var marginX = 10;
    var marginY = 30;
    var chartWidth = 300;
    var chartHeight = 150;

    var path = null;

    var graphicXaxis = null, graphicYaxis = null;

    document.addEventListener('DOMContentLoaded', function() {

        var chart = d3.select("svg")
            .attr("width", chartWidth)
            .attr("height", chartHeight)
            .attr("overflow","visible")
            .append("g");

        graphicYaxis = d3.select("svg").append("g")
            .attr("class", "y axis")
            .attr("transform", "translate("+marginX/2+",0)");

        graphicXaxis = d3.select("svg").append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0,"+(-marginY/2)+")");

        path = chart.append("path")
            .attr("class", "chartPath")
            .style("stroke-width", 2)
            .style("stroke-opacity", 0.7)
            .style("stroke", "#904b58")
            .style("fill", "none");

    }.bind(this));


    this.update = function(fromYear, toYear, data){

        if(path == null){
            return;
        }
        var minX = fromYear;
        var maxX = toYear;
        var minY = 0;
        var maxY = 0;

        for(var i = 0; i<data.length; i++){
            maxY = Math.max(maxY, data[i][1]);
        }

        var xScaler = d3.scaleLinear()
            .domain([minX, maxX])
            .range([marginX,chartWidth]);

        var yScaler = d3.scaleLinear()
            .domain([minY, maxY])
            .range([chartHeight - marginY,0]);

        var yAxis = d3.axisRight().scale(yScaler).ticks(3).tickFormat(function(d){
            return d;
        });

        var xAxis = d3.axisTop().scale(xScaler).ticks(3).tickFormat(function(d){
            return d;
        });

        graphicYaxis.call(yAxis);
        graphicXaxis.call(xAxis);


        var line = d3.line()
            .x(function(d){
                return xScaler(d[0]);
            })
            .y(function(d){
                return yScaler(d[1]);
            });

        path.attr("d", line(data));

    }.bind(this);


};







function animateCrash(dep,arr,crash,targetSVG, planeSVG,depLabel, arrLabel, crashDescription,map,crashId){
    /**
     * Animate single crash
     */
    return new Promise(function(resolve,reject){

        var arc = -0.8;
        var shadowAlpha = 0.3;
        var planeScale = 0.05;
        var dashLength = 2;
        var flightLineAlpha = 0.5;
        var initialImgAlpha = 1;
        var afterAnimImgAlpha = 0.5;
        var positionScale = 2;
        var flightLineArc = {
            "id": "flightLineArc"+crashId,
            "dashLength": dashLength,
            //"arc": arc,
            "alpha": flightLineAlpha,
            "arrowAlpha": flightLineAlpha,
            "latitudes": [dep[0],arr[0]],
            "longitudes": [dep[1],arr[1]]
        };

        var crashLine = {
            "id": "crashLine"+crashId,
            "arrowAlpha":0,
            "color":'#FF0000',
            "alpha": 0,
            "latitudes": [dep[0],crash[0]],
            "longitudes": [dep[1],crash[1]]
        };
        var crashLineArc = {
            "color":'#FF0000',
            "id": "crashLineArc"+crashId,
            "alpha": 0,
            "arrowAlpha":0,
            "arc": arc,
            "latitudes": [dep[0],crash[0]],
            "longitudes": [dep[1],crash[1]]
        };

        /*map["dataProvider"]["lines"].push(crashLine);
        map["dataProvider"]["lines"].push(crashLineArc);*/

        // Create images (with and without animation)
        var crashImg=  {
            "id": "crash"+crashId,
            "imageURL":'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Skull_%26_crossbones.svg/513px-Skull_%26_crossbones.svg.png',
            "title": crashDescription,
            "latitude": crash[0],
            "longitude": crash[1],
            "rollOverScale":3,
            "alpha":initialImgAlpha
            //"linkToObject": ["arrTarget"+crashId,"depTarget"+crashId]
        };

        var depTarget=  {
            "id": "depTarget"+crashId,
            "svgPath": targetSVG,
            "title": depLabel,
            "latitude": dep[0],
            "longitude": dep[1],
            "rollOverScale":2,
            "alpha":initialImgAlpha,
            "LinkToObject":"crash"+crashId
        };
        var arrTarget=  {
            "id": "arrTarget"+crashId,
            "svgPath": targetSVG,
            "title": arrLabel,
            "latitude": arr[0],
            "longitude":arr[1],
            "rollOverScale":2,
            "alpha":initialImgAlpha,
            "LinkToObject":"crash"+crashId
        };

        var crashPathShadow = {
            "svgPath": planeSVG,
            "positionOnLine": 0,
            "color": "#585869",
            "alpha": shadowAlpha,
            "animateAlongLine": true,
            "lineId": "crashLine"+crashId,
            "flipDirection": false,
            "loop": false,
            "scale": planeScale,
            "positionScale": positionScale
        };
        var flightCrash = {
            "svgPath": planeSVG,
            "positionOnLine": 0,
            "color": "#a20819",
            "alpha": 0.7,
            "animateAlongLine": true,
            "lineId": "crashLineArc"+crashId,
            "flipDirection": false,
            "loop": false,
            "scale": planeScale,
            "positionScale": positionScale
        };

        /*var crash_descriptor = {
            id:crashId,
            lines: lines.slice(),
            images: images.slice()
        };*/


        var animDuration = map["imagesSettings"]["animationDuration"]*1500;


        map["dataProvider"]["lines"].push(flightLineArc,crashLine,crashLineArc);
        map["dataProvider"]["images"].push(depTarget,arrTarget,crashImg,flightCrash,crashPathShadow);
        setTimeout(function(){
            //if(map["dataProvider"]["lines"].length>3)
            map.validateData();
            map["dataProvider"]["images"].pop();
            map["dataProvider"]["images"].pop();
            var len =  (map["dataProvider"]["images"]).length;
            map["dataProvider"]["images"][len-1]["alpha"]= afterAnimImgAlpha;
            map["dataProvider"]["images"][len-2]["alpha"]= afterAnimImgAlpha;
            map["dataProvider"]["images"][len-3]["alpha"]= afterAnimImgAlpha;
            resolve();
        },animDuration);

    });
}


/**
 * Created by noodle on 10.12.17.
 */


/**  chartMap
 * id#
 * id#
 * Date#
 * Route#
 * Type#
 * LatLongCrash#
 * Start City Crash#
 * End City Crash#
 * Start City Geo Crash#
 * End City Geo Crash#
 * To_show#
 * Operator#
 * Summary#
 * Aboard#
 * Fatalities
 *
 *
 * id#
 * Date#
 * Time#
 * Location#
 * Operator#
 * Route#
 * Type#
 * Aboard#
 * Fatalities#
 * Summary#
 * LatLongCrash#
 * Start City Crash#
 * End City Crash#
 * Start City Geo Crash#
 * End City Geo Crash#
 * To_show
 */

var ChartMapParser = new function(){

    function parsePosition(latLongString){
        var resStr = latLongString.substr(1,latLongString.length-1).split(",");
        return {
            lat: parseFloat(resStr[0]),
            long:parseFloat(resStr[1])
        };
    }

    /**
     id#Year#Country#city#latitude#longitude#AttackType#Killed#Wounded#Group

     */
    this.lineParser = function(line){

        var cols = line.split("#");

        console.log("parsed");

        return {
            id:parseInt(cols[0]),
            company:cols[9],
            dep_lat:parseFloat(cols[4]),
            dep_long:parseFloat(cols[5]),
            arr_lat:parseFloat(cols[4]),
            arr_long:parseFloat(cols[5]),
            crash_lat:parseFloat(cols[4]),
            crash_long:parseFloat(cols[5]),
            startCity:"",
            endCity:"",
            description:cols[9],
            nb_a_bord:"",
            nb_mort:parseInt(cols[7]),
            year:parseInt(cols[1])
        };
    };

    this.filter = function(line){
        return true;
    };

};



/**  chartLine
 id#Year#Country#city#latitude#longitude#AttackType#Killed#Wounded#Group
 */

var LineChartParser = new function(){

    this.lineParser = function(line){

        var cols = line.split("#");

        console.log("year : "+cols[1]);
        var res = {
            id:cols[0],
            nb_mort:parseInt(cols[7]),
            company:cols[9],
            year:parseInt(cols[1])
        };

        return res;
    };

    this.filter = function(line){
        return true;
    };
};



function load_data(filePath, lineParser, filter){



    return new Promise(function(resolve, error){

        var rawFile = new XMLHttpRequest();
        rawFile.open("GET", filePath, true);

        rawFile.onreadystatechange = function() {

            if (rawFile.readyState == 4) {


                var allText = rawFile.responseText;
                var lines = allText.split("\n");


                var objects = [];
                for(var i = 1; i<lines.length; i++){

                    if(filter(lines[i])){
                        objects.push(lineParser(lines[i]))
                    }
                }

                resolve(new function(){

                    var allData = objects;

                    this.data = allData;

                    this.getData = function(fromYear, toYear){
                        var requiredData = [];
                        for(var i = 0; i<allData.length; i++){
                            if(fromYear <= allData[i].year && allData[i].year <= toYear){
                                requiredData.push(allData[i]);
                            }
                        }
                        return requiredData;
                    }.bind(this);


                });

            }

        };

        rawFile.send();
    });

}



function ChartLineData(objects){


    var allData = objects;

    this.data = allData;

    this.getData = function(fromYear, toYear){

        var requiredData = [];

        for(var i = 0; i<allData.length; i++){
            if(fromYear <= allData[i].year && allData[i].year <= toYear){
                requiredData.push(allData[i]);
            }
        }

        return requiredData;

    }.bind(this);

    // TODO :
    var yearAndDeath = (
        function(){  
            var tmp = {};  
            for(var i = 0; i<objects.length; i++){ 
                if(!(objects[i].year in tmp)){ 
                    tmp[objects[i].year] = 0; 
                }

                tmp[objects[i].year] += 1;

            }  

            var yearAndDeath = [];  
            for(var key in tmp){ 
                yearAndDeath.push([parseInt(key), tmp[key]]); 
            }   

            return yearAndDeath; 

        })();


      this.getYearAndCrashes = function(fromYear, toYear){ 
        return yearAndDeath.filter(t => fromYear <=  t[0] && t[0]<=toYear); 
    }.bind(this); 

    this.companyAndDeathSorted = function(fromYear, toYear){  

        var data = this.getData(fromYear, toYear); 
        var crashCompanyCounter = {};  
        for(var i = 0; i<data.length; i++){ 
            if(!(data[i].company in crashCompanyCounter)){ 
                crashCompanyCounter[data[i].company] = 0; 
            } 

            if(!isNaN(data[i].nb_mort)){
                crashCompanyCounter[data[i].company] += data[i].nb_mort; 
            }
        }  

        var tmp = [];  

        for(var key in crashCompanyCounter){ 

            tmp.push([key, crashCompanyCounter[key]]); 
        }  

        crashCompanyCounter = tmp; 

        crashCompanyCounter.sort(function(e1,e2){ return e2[1]-e1[1];});  


        return crashCompanyCounter; 

    }.bind(this);


}



function visualiseData(data,map,targetSVG,planeSVG){
    /**
     * visualise the all the crashes
     */
    return new Promise(function(resolve,reject){
        if (data.length==0) {
            return;
        }
        var crash = data.shift();
        var dep = [crash["dep_lat"],crash["dep_long"]];
        var arr = [crash["arr_lat"],crash["arr_long"]];
        var crashPos = [crash["crash_lat"],crash["crash_long"]];
        var depLabel= crash.startCity;
        var arrLabel = crash.endCity;
        var crashDescription= crash["description"];
        var crashId = crash['id'];
        animateCrash(dep,arr,crashPos,targetSVG, planeSVG,depLabel, arrLabel, crashDescription,map,crashId)
            .then(function(res,err){
                visualiseData(data,map,targetSVG,planeSVG);
            }) ;
    });
}


function init_range_selector(begin,end){


    $( function() {
        $( "#slider-range" ).slider({
            range: true,
            min: begin,
            max: end,
            animate: true,
            values: [ begin, end ],
            slide: function( event, ui ) {


                if(dataLoader == null){
                    return;
                }

                data=dataLoader.getData(ui.values[0], ui.values[1]);

                var companySorted = chartLineData.companyAndDeathSorted(ui.values[0], ui.values[1]);
                CrashChart.update(ui.values[0], ui.values[1], chartLineData.getYearAndCrashes(ui.values[0],ui.values[1]));
                sideMenuManager.updateResult(companySorted);

                document.querySelector(".yearRangeTable .fromValue").innerHTML = ui.values[0];
                document.querySelector(".yearRangeTable .toValue").innerHTML = ui.values[1];


            }.bind(this)
        });

        $( "#amount" ).val(   $( "#slider-range" ).slider( "values", 0 ) +
            " - " + $( "#slider-range" ).slider( "values", 1 ) );

    } );

}




function showCrashes(data){

    map["dataProvider"]["images"]=[];

    var arrayLength = data.length;
    for (var i = 0; i < arrayLength; i++) {

        var crash = data[i];
        var dep = [crash["dep_lat"],crash["dep_long"]];
        var arr = [crash["arr_lat"],crash["arr_long"]];
        var crashPos = [crash["crash_lat"],crash["crash_long"]];
        var depLabel= crash.startCity;
        var arrLabel = crash.endCity;
        var crashDescription= crash["description"];
        var crashId = crash['id'];
        staticCrash(dep,arr,crashPos,targetSVG, planeSVG,depLabel, arrLabel, crashDescription,map,crashId);
        //Do something
    }

    map.validateData();

}

function staticCrash(dep,arr,crash,targetSVG, planeSVG, depLabel, arrLabel, crashDescription,map,crashId) {

    var crashImg =  {
        "id": "crash"+crashId,
        "imageURL":'crash.png',
        "title": crashDescription,
        "latitude": crash[0],
        "longitude": crash[1],
        "rollOverScale":1.5,
        "alpha":1,
        "scale":2,
        "fixedSize":false
    };

    map["dataProvider"]["images"].push(crashImg);

}














/**
 * Created by noodle on 14.11.17.
 */

/**
 * Created by noodle on 26.10.17.
 */




function animate(elem, property, fromValue, toValue, unit, duration){


    var frameDuration = 20;

    var totalFrames = duration/frameDuration;

    elem.style[property] = fromValue +unit;


    return new Promise(function(resolve, reject) {

        var currFrame = 0;


        var mon = setInterval(function(){

            var rate = currFrame/totalFrames;
            var newValue = rate*toValue + (1-rate)*fromValue;

            elem.style[property] = newValue+unit;


            if(currFrame>=totalFrames){

                elem.style[property] = toValue+unit;
                clearInterval(mon);
                resolve();
            }

            currFrame+=1;

        },frameDuration);


    });


}




function titleAnim(title, newTitleName, backgroundColor, duration){

    var txtWrapper = document.createElement("span");
    txtWrapper.appendChild(document.createTextNode(newTitleName));

    var colorWrapper = document.createElement("div");


    var paddingTitleTop = parseInt(window.getComputedStyle(title).paddingTop);
    var paddingTitleBottom = parseInt(window.getComputedStyle(title).paddingBottom);

    title.style.paddingTop = "0px";
    title.style.paddingBottom="0px";

    colorWrapper.style.position="absolute";
    colorWrapper.style.width="0px";
    colorWrapper.style.backgroundColor=backgroundColor;
    colorWrapper.style.height=title.clientHeight+"px";

    console.log(paddingTitleTop);
    title.style.paddingTop = paddingTitleTop+"px";
    title.style.paddingBottom = paddingTitleBottom+"px";


    colorWrapper.style.width="0px";
    colorWrapper.style.marginLeft = "0px";

    animate(title, "opacity", 1, 0, "", duration)
        .then(function(res){

            title.style.opacity ="1";
            txtWrapper.style.opacity="0";
            while (title.firstChild) {
                title.removeChild(title.firstChild);
            }
            title.appendChild(colorWrapper);
            title.appendChild(txtWrapper);

            return animate(colorWrapper, "width", 0, title.clientWidth , "px", duration/2);
        })
        .then(function(res){
            txtWrapper.style.opacity="1";

            animate(colorWrapper, "margin-left", 0, title.clientWidth, "px", duration/3)

            return animate(colorWrapper, "width", title.clientWidth, 0, "px", duration/2);
        });



};

