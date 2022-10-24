// cuPoly - A GPGPU-based library for doing polynomial arithmetic on RLWE-based cryptosystems
// Copyright (C) 2017-2021, Pedro G. M. R. Alves - pedro.alves@ic.unicamp.br

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#ifndef LOG_H
#define LOG_H
#include <iostream>
#include <string>

enum log_mode {QUIET, INFO, WARNING, DEBUG, ERROR, VERBOSE};

/**
 * \brief      Offers a simple mechanism for logging using std::cout.
 * 
 * The Logger class follows the Singleton design pattern.
 * Only those messages tagged with a level lower or equal to
 * the selected mode will be print to log. The supported levels are defined in #log_mode
 * 
 * - QUIET: Nothing shall be written to log.
 * - INFO: Level 1
 * - WARNING: Level 2
 * - DEBUG: Level 3
 * - ERROR: Level 4
 * - VERBOSE: Level 5
 */
class Logger{
    private:
        /* Here will be the instance stored. */
        static Logger* instance;
        int mode;

        /* Private constructor to prevent instancing. */
        Logger(){
            mode = QUIET;
        };

        void __inline__ print(const char *s, int logname);

    public:
        
        /**
         * @brief      This is the way to obtain an instance of this Singleton
         *
         * @return     The instance.
         */
        static Logger* getInstance(){
            if (!instance)
              instance = new Logger;
            return instance;
        }

        void set_mode(int m){
            mode = m;
        }

        void log_debug(const char* s);
        void log_info(const char *s);
        void log_warning(const char* s);
        void log_error(const char* s);
};
#endif