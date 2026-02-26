#pragma once

#include "op_desc.h"
#include <string>

// Parse a .moe config file (key=value format)
// Returns an OpDesc populated with MoE configuration
OpDesc parse_moe_config(const std::string& filename);
