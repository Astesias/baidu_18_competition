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
            order='run'
            break
        case 2:
            order='update'
            location.reload();
            break
        case 3:
            order='exit'
            break
        case 4:
            order='video_on'
            break
        case 5:
            order='video_off'
            break
        case 7:
            order='shot'
            break
        case 8:
            order='shot_del'
            break
        case 9:
            order='shot_off'
            break
        default:
            order=index

    }

    $("#list-o").text(order)
    $.post('/order/',{'order':order},dataType="json")
}




var d_maxlength_=10
var d_maxlength=10;
var s_maxlength_=10
var s_maxlength=10;
var getdata=null




function msd_d(msg){
    list=$("#list-d");
    var varitem= "<div id=d{0} class=\"log\">{1}</div>"
    var fmtitem= String.format(varitem,d_maxlength,msg)
    list.append(fmtitem);
    d_maxlength--;
    if (d_maxlength<0){
        var rmitem= String.format("#d{0}",d_maxlength+d_maxlength_+1)
        $(rmitem).remove()
    
    var ele = document.getElementById('list-d');
    ele.scrollTop = ele.scrollHeight;
    }
}

function msd_s(msg){
    list=$("#list-s");
    var varitem= "<div id=s{0} class=\"log\">{1}</div>"
    var fmtitem= String.format(varitem,s_maxlength,msg)
    list.append(fmtitem);
    s_maxlength--;
    if (s_maxlength<0){
        var rmitem= String.format("#s{0}",s_maxlength+s_maxlength_+1)
        $(rmitem).remove()
    }

    var ele = document.getElementById('list-s');
    ele.scrollTop = ele.scrollHeight;
}




function ADD(){

    list=$("#list-d");
    list=$("#list-s");

    $.get('/data'+ Date.now(),{},
                function(data){
                    if (data!='Nodata')
                        getdata=data
                    else
                        getdata=null
                })
    if (getdata==null || getdata=='NoData')
            return

    console.log(getdata,getdata.slice(1))
    if (getdata.indexOf('D')!=-1){
        msd_d(getdata.slice(1))
        return
    }
    if (getdata.indexOf('S')!=-1){
        msd_s(getdata.slice(1))
        return
    }


}





setInterval(ADD, 1000);