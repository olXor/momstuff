#ifndef EVENT_HEADER_FILE
#define EVENT_HEADER_FILE

#include <ncurses.h>
#include <stdio.h>
#include <string.h>
//#include "nvwa/debug_new.h"

#define NUM_EVENTS 100

class EventLog {
    public:

    void initializeEventLog();
    void addEvent(const char *event, WINDOW *eventWindow);
    void writeEvents(WINDOW *window);
    EventLog() { initializeEventLog(); };
    ~EventLog() {
        for(int i=0; i<NUM_EVENTS; i++) {
            delete [] eventLog[i];
        }
    };

    private:

    const char *eventLog[NUM_EVENTS];
    int messageLength[NUM_EVENTS];
};

#endif
