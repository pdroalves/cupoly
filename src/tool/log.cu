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


#include <cuPoly/tool/log.h>

Logger *Logger::instance = 0;

void Logger::log_debug(const char* s){
    print(s, DEBUG);
}
void Logger::log_info(const char* s){
    print(s, INFO);
}
void Logger::log_warning(const char* s){
    print(s, WARNING);
}
void Logger::log_error(const char* s){
    print(s, ERROR);
}

void __inline__ Logger::print(const char* s, const int logname){
    if(logname <= mode)
        std::cout << s << std::endl;
}