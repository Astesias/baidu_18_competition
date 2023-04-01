String.format = function() {
    if (arguments.length == 0)
        return null;
    var str = arguments[0];
    for ( var i = 1; i < arguments.length; i++) {
        var re = new RegExp('\\{' + (i - 1) + '\\}', 'gm');
        str = str.replace(re, arguments[i]);
    }
    return str;
}

function print(arg){
    console.log(arg);
}

function button(index){
    switch(index){
        case 1:
            order='start'
            break
        default:
            order=index

    }

    $("#list-o").text(order)
    $.post('/order/',{'order':order},dataType="json")
}




var maxlength_=10
var maxlength=10;
var getdata=null
function ADD(){

    list=$("#list-d");

    var varitem= "<div id={0} class=\"log\">{1}</div>"

    $.get('/data'+ Date.now(),{},
                function(data){
                    if (data!='Nodata')
                        getdata=data
                    else
                        getdata=null
                })
    if (getdata==null || getdata=='NoData')
            return

    var fmtitem= String.format(varitem,maxlength,getdata)

    list.append(fmtitem);
    
    var ele = document.getElementById('list-d');
    ele.scrollTop = ele.scrollHeight;

    maxlength--;

    if (maxlength<0){
        var rmitem= String.format("#{0}",maxlength+maxlength_+1)
        $(rmitem).remove()
    }
}

setInterval(ADD, 1000);