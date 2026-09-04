// A safe C++ function for the CI smoke test: it only mentions "strcpy" inside
// a comment, so Layer 2 (AST validation) must REJECT that regex match. The
// pipeline verdict should therefore be "likely_safe".
#include <cstdio>
#include <string>

void greet(const std::string &name) {
    // Avoid strcpy(): use std::string which manages its own memory.
    std::string msg = "hello, " + name;
    std::printf("%s\n", msg.c_str());
}
