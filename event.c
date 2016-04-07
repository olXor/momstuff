#include "event.h"
#include <stdlib.h>

void EventLog::initializeEventLog() {
    int i;
    for(i=0; i<NUM_EVENTS; i++) {
        eventLog[i]='\0';
        messageLength[i]=0;
    }
}

void EventLog::addEvent(const char *event, WINDOW *window) {
    if(!eventLog[NUM_EVENTS-1]=='\0')
        delete [] eventLog[NUM_EVENTS-1];
        //free((char*)eventLog[NUM_EVENTS-1]);

    int i;
    for(i=NUM_EVENTS-1; i>0; i--) {
        eventLog[i]=eventLog[i-1];
        messageLength[i]=messageLength[i-1];
    }

    char* ecpy = new char[5000];
    strcpy(ecpy, event);
    eventLog[0] = ecpy;
    messageLength[0] = strlen(ecpy);

    //display the new events list in the given window
    writeEvents(window);
}

void EventLog::writeEvents(WINDOW *window) {
    werase(window);
    int maxy, maxx;
    getmaxyx(window, maxy, maxx);
    int line = maxy-1;
    int numlines;
    int curline;
    for(int i=0;i<NUM_EVENTS;i++) {
        //round up
        numlines = ((int)((messageLength[i]+maxx-1)/maxx));
        line-=numlines;
        if(line<0)
            break;
        curline=0;
        for(int j=0;j<messageLength[i];j++) {
            if(j%maxx==0)
                curline++;
            mvwaddch(window,line+curline,(int)j%maxx,eventLog[i][j]);
        }
    }
    wrefresh(window);
}
