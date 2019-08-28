function addLoadEvent(func) {
  var oldonload = window.onload;
  if (typeof window.onload != 'function') {
    window.onload = func;
  } else {
    window.onload = function() {
      if (oldonload) {
        oldonload();
      }
      func();
    }
  }
}


function tabber(){
    let navTabs = document.getElementsByClassName("nav-tabs");
    let selectAll = function(ind){
        for(let navTab of navTabs){
            let dom = navTab.childNodes[ind];
            let old = dom.onclick;
            dom.onclick = null;
            dom.click();
            dom.onclick = old;
        }
    }
    let registerAll = function(){
        for(let navTab of navTabs){
            let commandLineTab = navTab.childNodes[0];
            let pythonTab = navTab.childNodes[1];
            commandLineTab.onclick = function(){
                selectAll(0);
            }
            pythonTab.onclick = function(){
                selectAll(1);
            }
        }
    }
    registerAll();
};

addLoadEvent(tabber);
