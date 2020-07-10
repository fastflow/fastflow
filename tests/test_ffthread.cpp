#include <string>
#include <ff/node.hpp>
#include <iostream>

class manager:public ff::ff_thread {
public:

    manager():running(false) {}

    void enterActive() {
    if (!running)
    {
        running = true;
        freeze();
        spawn();
        freeze();
        wait_freezing();
    }
    
    thaw();
}
    
    void enterActive_2() {
	if (!running)
	    {
		running = true;
		freeze();
		spawn();
		return;   //esco perche' non serve fare la thaw() dopo la spawn
	    }
	wait_freezing();  // added
	thaw();           // added
    }
    
    void enterPassive() {
	freeze();
    }
    
    void* svc(void*)
    {
	if (!running)
	    {
		stop();
		return EOS_NOFREEZE;
	    }
	
	usleep(1000);
	
	return GO_ON;
    }

private:
    bool running;
      
};





int main() {
    manager man;
    

    man.enterActive_2();
    usleep(600 * 1000);
    man.enterPassive();
    usleep(700 * 1000);
    man.enterActive();
    man.wait();
    
    std::cout<<"Fine"<<std::endl;
    
    return 0;
}
