
function draw(dataset) {
  var width = 600, height = 300;
			 
  var main = d3.select('.container svg').append('g')
   .classed('main', true)
   .attr('transform', "translate(" + width/2 + ',' + height/2 + ')');
  
  // transfer dataset 
  var pie = d3.layout.pie()
   .sort(null)
   .value(function(d) {
    return d.value;
   });
  // pie is a function
  var pieData = pie(dataset);
  // compute arc path
  var radius = 100;
  var arc = d3.svg.arc()
   .innerRadius(0)
   .outerRadius(radius);
  var outerArc = d3.svg.arc()
   .innerRadius(1.2 * radius)
   .outerRadius(1.2 * radius);
  var oArc = d3.svg.arc()
   .innerRadius(1.1 * radius)
   .outerRadius(1.1 * radius);
  var slices = main.append('g').attr('class', 'slices');
  var lines = main.append('g').attr('class', 'lines');
  var labels = main.append('g').attr('class', 'labels');
  // add element in arc
  var arcs = slices.selectAll('g')
   .data(pieData)
   .enter()
   .append('path')
   .attr('fill', function(d, i) {
    return getColor(i);
   })
   .attr('d', function(d){
    return arc(d);
   });
  // add text label
  var texts = labels.selectAll('text')
   .data(pieData)
   .enter()
   .append('text')
   .attr('dy', '0.35em')
   .attr('fill', function(d, i) {
    return getColor(i);
   })
   .text(function(d, i) {
    return d.data.name;
    //return [d.data.name, d.data.value];
   })
   .style('text-anchor', function(d, i) {
    return midAngel(d)<Math.PI ? 'start' : 'end';
   })
   .attr('transform', function(d, i) {
    // find the center 
    var pos = outerArc.centroid(d);
    // change text x axis
    pos[0] = radius * (midAngel(d)<Math.PI ? 1.5 : -1.5);
 
    return 'translate(' + pos + ')';
   })
   .style('opacity', 1);
 
  var polylines = lines.selectAll('polyline')
   .data(pieData)
   .enter()
   .append('polyline')
   .attr('points', function(d) {
    return [arc.centroid(d), arc.centroid(d), arc.centroid(d)];
   })
   .attr('points', function(d) {
    var pos = outerArc.centroid(d);
    pos[0] = radius * (midAngel(d)<Math.PI ? 1.5 : -1.5);
    return [oArc.centroid(d), outerArc.centroid(d), pos];
   })
   .style('opacity', 0.5);
  };
  function midAngel(d) {
  return d.startAngle + (d.endAngle - d.startAngle)/2;
  }
  function getColor(idx) {
  var palette = [
   '#2ec7c9', '#b6a2de', '#5ab1ef', '#ffb980', '#d87a80',
   '#8d98b3', '#e5cf0d', '#97b552', '#95706d', '#dc69aa',
   '#07a2a4', '#9a7fd1', '#588dd5', '#f5994e', '#c05050',
   '#59678c', '#c9ab00', '#7eb00a', '#6f5553', '#c14089'
  ]
  return palette[idx % palette.length];
			  }