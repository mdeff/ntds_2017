/*******************************************************************************************************************************************
Preliminary Task: Set Window Parameters

      // Get dimension of the browser window
      // adjust the browser window aize accoedingly
**********************************************************************************************************************************************/

var width_adjust = 20; // adjust this parameter to vary window width
var height_adjust = 100; // adjust this parameter to vary window heoght
const DIMENSIONS = getWindowDimensions(); // Call the getWindowDimensions window function to get window size parameters. The function getWindowDimensions
                    // is defined at the last part of this page
// Padding.
const PADDING = {'left': 0, 'right': 0, 'top': 20, 'bottom': 20};
const WIDTH = DIMENSIONS.width;    // substract or adjust returned window width
const HEIGHT = DIMENSIONS.height;  //substract or adjust returned window heoght
const SCALE0 = (WIDTH - 1) / 2 / Math.PI;



const PATH = "./data1/"
var year = '1960'
var category = 5
//BILGUUN
var label = 2
//BILGUUN
//const colormaps=[//colormap that contains different color paletts for each category
//    ['rgb(247,244,249)','rgb(231,225,239)','rgb(212,185,218)','rgb(201,148,199)','rgb(223,101,176)','rgb(231,41,138)','rgb(206,18,86)','rgb(152,0,67)','rgb(103,0,31)'],
//    ['rgb(255,247,243)','rgb(253,224,221)','rgb(252,197,192)','rgb(250,159,181)','rgb(247,104,161)','rgb(221,52,151)','rgb(174,1,126)','rgb(122,1,119)','rgb(73,0,106)'],
//    ['rgb(255,255,229)','rgb(247,252,185)','rgb(217,240,163)','rgb(173,221,142)','rgb(120,198,121)','rgb(65,171,93)','rgb(35,132,67)','rgb(0,104,55)','rgb(0,69,41)'],
//    ['rgb(255,255,217)','rgb(237,248,177)','rgb(199,233,180)','rgb(127,205,187)','rgb(65,182,196)','rgb(29,145,192)','rgb(34,94,168)','rgb(37,52,148)','rgb(8,29,88)'],
//    ['rgb(255,255,229)','rgb(255,247,188)','rgb(254,227,145)','rgb(254,196,79)','rgb(254,153,41)','rgb(236,112,20)','rgb(204,76,2)','rgb(153,52,4)','rgb(102,37,6)'],
//    ['rgb(255,255,204)','rgb(255,237,160)','rgb(254,217,118)','rgb(254,178,76)','rgb(253,141,60)','rgb(252,78,42)','rgb(227,26,28)','rgb(189,0,38)','rgb(128,0,38)'],
//    ['rgb(255,247,236)','rgb(254,232,200)','rgb(253,212,158)','rgb(253,187,132)','rgb(252,141,89)','rgb(239,101,72)','rgb(215,48,31)','rgb(179,0,0)','rgb(127,0,0)'],
//]
//var palette = colormaps[0]


d3.select("#slider-time")
  .on("change", function() { 
        year=this.value;
        d3.selectAll("svg")
          .remove(); 
        update();
})


d3.select("#categories")
  .on("change", function() { 
        category=this.value;
        d3.selectAll("svg")
          .remove(); 
        update()
})
//BILGUUN
d3.select("#labels")
  .on("change", function() { 
        label=this.value;
        d3.selectAll("svg")
          .remove(); 
        update()
})

function UrlExists(url)
{
    var http = new XMLHttpRequest();
    http.open('HEAD', url, false);
    http.send();
    return http.status!=404;
}
//BILGUUN
//function listner that update the dashboard
function update(){
    console.log(year)
    var cat=document.getElementById("categories").options[category-1].text
    var lab=document.getElementById("labels").options[label-1].text
    if(lab=='Income') lab='incomeLevel' 
    else lab='region'

    var data='data/'+cat.toLowerCase()+year+'-'+(parseInt(year)+2)+lab+'.json'
    var exist=UrlExists(data)
    if(lab=='incomeLevel' && !exist){
        var data='data/'+cat.toLowerCase()+year+'-'+(parseInt(year)+2)+'.json'
        exist=UrlExists(data)
    }
    console.log([exist,data])
    if(exist){
        d3.json(data, function(error, graph_data) {
            if (error) {
                console.log(error)
            } else {
                draw(graph_data);
            }
        }); 
    } else {
        var svg =  d3.select("#chart").append("svg")
                  .attr("width",WIDTH) 
                  .attr("height",HEIGHT+ PADDING.top + PADDING.bottom) 
                  .attr("class","canvas") 
        
        

        var catlabel = svg.append("text")
           .attr("class", "year label")
           .attr("text-anchor", "start")
           .attr("y", HEIGHT - 250)
           .attr("x", 200)
           .text("Sorry, Not Available...");
    }
} 

window.onload = function() {
  update()
};

let draw = (graph_data) =>{
        var svg =  d3.select("#chart").append("svg")
                        .attr("width",WIDTH) 
                        .attr("height",HEIGHT+ PADDING.top + PADDING.bottom) 
                        .attr("class","canvas")
        //add encompassing group for the zoom 
        var g = svg.append("g")
                   .attr("class", "everything");
        //add zoom capabilities 
        var zoom_handler = d3.zoom()
            .on("zoom", zoom_actions);

        zoom_handler(svg);

        //Zoom functions 
        function zoom_actions(){
            g.attr("transform", d3.event.transform)
        }
        
        var color = d3.scaleOrdinal(d3.schemeCategory20);
                
        var simulation = d3.forceSimulation()
                           .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(100).strength(0.1))
                           .force("charge", d3.forceManyBody())
                           .force("center", d3.forceCenter(WIDTH / 2, HEIGHT / 2 -10))
                           //.force("gravity", 0) 
                           //.force("y", d3.forceY(1))
                           //.force("x", d3.forceX(1))
                           //.force("distance", 100);
      
        var link = svg.append("g")
                      .attr("class", "links")
                      .selectAll("line")
                      .data(graph_data.links)
                      .enter().append("line")
                      .attr("stroke", "#999")
                      .attr("stroke-width", function(d) { return Math.sqrt(d.value); })
                    
        
        var node = svg.append("g")
                      .attr("class", "nodes")
                      .selectAll("circle")
                      .data(graph_data.nodes)
                      .enter().append("circle")
                      .attr("r", 10)
                      .attr("fill", function(d) { return color(d.group); })
                      .call(d3.drag()
                      .on("start", dragstarted)
                      .on("drag", dragged)
                      .on("end", dragended))
                      .on("mouseover", mouseover)
                      .on("mouseout", mouseout)
        
      
        node.append("title")
            .text(function(d) { return d.id; });    
      
        simulation.nodes(graph_data.nodes)
                  .on("tick", ticked);
        simulation.force("link")
                  .links(graph_data.links);    
        
        //var animating = true;
      
        // We'll also define a variable that specifies the duration
        // of each animation step (in milliseconds).
      
        //var animationStep = 400;
        

        function ticked() {
        
      
          link//.transition().ease('linear').duration(animationStep)
              .attr("x1", function(d) { return d.source.x; })
              .attr("y1", function(d) { return d.source.y; })
              .attr("x2", function(d) { return d.target.x; })
              .attr("y2", function(d) { return d.target.y; });
      
          node//.transition().ease('linear').duration(animationStep)
              .attr("cx", function(d) { return d.x; })
              .attr("cy", function(d) { return d.y; });
        }

        function drawLink(d) {
          context.moveTo(d.source.x, d.source.y);
          context.lineTo(d.target.x, d.target.y);
        }

        function drawNode(d) {
          context.moveTo(d.x + 3, d.y);
          context.arc(d.x, d.y, 3, 0, 2 * Math.PI);
        }

        function dragsubject() {
          return simulation.find(d3.event.x, d3.event.y);
        }
      
        function dragstarted(d) {
              if (!d3.event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
        }
      
        function dragged(d) {
              d.fx = d3.event.x;
              d.fy = d3.event.y;
        }
      
        function dragended(d) {
              if (!d3.event.active) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
        }
      
        function mouseover(d){
              tooltip.transition()
                     .duration(500)
                     .style("opacity", 9);
              console.log("d", d)
              tooltip.html(d.id)
                     .style("left", (d3.event.pageX) + "px")
                     .style("top", (d3.event.pageY - 50) + "px");
        }
      
        function mouseout(){
              tooltip.transition()
                     .duration(500)
                     .style("opacity", 0);
        }
      
          //BILGUUN
          var catlabel = svg.append("text")
             .attr("class", "category label")
             .attr("text-anchor", "start")
             .attr("y", HEIGHT - 250)
             .attr("x", 100)
             .text(document.getElementById("categories").options[category-1].text +" - "+ document.getElementById("labels").options[label-1].text);
      
          var yearlabel = svg.append("text")
             .attr("class", "year label")
             .attr("text-anchor", "start")
             .attr("y", HEIGHT - 150)
             .attr("x", 100)
             .text(year+'-'+(parseInt(year)+2));
          //BILGUUN
}

function getWindowDimensions() {

    var width =850;
    var height = 500;
    if (document.body && document.body.offsetWidth) {

        width = document.body.offsetWidth;
        height = document.body.offsetHeight;
    }

    if (document.compatMode == 'CSS1Compat' && document.documentElement && document.documentElement.offsetWidth) {

        width = document.documentElement.offsetWidth;
        height = document.documentElement.offsetHeight;
    }

    if (window.innerWidth && window.innerHeight) {

        width = window.innerWidth;
        height = window.innerHeight;
    }

    return {'width': width, 'height': height};
}

var tooltip = d3.select("section")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0); 
