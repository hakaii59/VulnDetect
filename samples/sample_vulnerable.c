// A deliberately-vulnerable C function: classic unbounded strcpy.
// Used as a CI smoke-test fixture: Layer 1 (regex) + Layer 2 (AST) must both
// confirm the strcpy call, so the pipeline verdict should be "vulnerable".
#include <stdio.h>
#include <string.h>

void copy_name(const char *input) {
    char name[16];
    strcpy(name, input);  // BAD: no bounds check -> CWE-120 buffer overflow
    printf("name = %s\n", name);
}
