#ifndef PARSE_HPP
#define PARSE_HPP

#include <iostream>
#include <sstream>

template<class T>
inline T parse(const char *s)
{
    T value;
    std::stringstream ss(s);
    ss >> value;
    
    if (ss.fail() || !ss.eof())
    {
        std::cerr << "Error in parse call: unable to convert string to the desired type" << std::endl;
    }
    
    return value;
}

#endif
