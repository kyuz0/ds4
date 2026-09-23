#include "ds4.h"

/* Standalone kernel tests do not link a loaded model engine. */
bool ds4_engine_is_glm53(ds4_engine *engine) {
    (void)engine;
    return false;
}
