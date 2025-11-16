import os
import sys
import re

DEFAULT_NUM_SAMPLES = 100000
DEFAULT_REGEX = "ex\\d+"


def parse_bound(bound):
    if "/" in bound:
        numerator, denominator = map(float, bound.split("/"))
        return numerator / denominator
    return float(bound)


def parse_c_file(filepath, func_regex):
    with open(filepath, "r") as file:
        content = file.read()

    pattern = re.compile(rf"(?s)(// ## PRE(?:.*?\n)+?)\s*([\w\s\*]+?)\s+({func_regex})\s*\(([^)]*)\)")

    matches = list(pattern.finditer(content))

    if not matches:
        exit(f"No functions found with the regex: {func_regex}")

    functions = []

    for match in matches:
        comments, return_type, func_name, params = match.groups()
        param_comments = re.findall(r"// ## PRE (\w+):\s*([-+.\d/]+),\s*([-+.\d/]+)", comments)
        bounds = {
            name: {
                "min": parse_bound(min_val),
                "max": parse_bound(max_val),
            }
            for name, min_val, max_val in param_comments
        }
        params = [param.strip() for param in params.split(",") if param.strip()]
        functions.append((func_name, bounds, params, return_type.strip()))

    return functions, content, matches


def create_driver_function(functions, num_samples_per_func):
    driver_code = [
        "#undef double",
        "",
        "#include <iostream>",
        "#include <random>",
        "#include <cstring>",
        "#include <chrono>",
    ]

    driver_code.append("#include <fstream>")
    driver_code.append("#include <limits>")
    driver_code.append("#include <iomanip>")
    driver_code.append("#include <string>")

    driver_code.append("")
    driver_code.append("int main(int argc, char* argv[]) {")
    driver_code.append('    std::string output_path = "";')
    driver_code.append("")
    driver_code.append("    for (int i = 1; i < argc; ++i) {")
    driver_code.append('        if (std::strcmp(argv[i], "--output-path") == 0) {')
    driver_code.append("            if (i + 1 < argc) {")
    driver_code.append("                output_path = argv[i + 1];")
    driver_code.append("                i++;")
    driver_code.append("            } else {")
    driver_code.append('                std::cerr << "Error: --output-path requires a path argument." << std::endl;')
    driver_code.append("                return 1;")
    driver_code.append("            }")
    driver_code.append("        }")
    driver_code.append("    }")
    driver_code.append("")
    driver_code.append("    bool save_outputs = !output_path.empty();")
    driver_code.append("")
    driver_code.append("    std::mt19937 gen(42);")
    driver_code.append("")

    driver_code.append("    std::ofstream ofs;")
    driver_code.append("    if (save_outputs) {")
    driver_code.append("        ofs.open(output_path);")
    driver_code.append("        if (!ofs) {")
    driver_code.append('            std::cerr << "Failed to open output file: " << output_path << std::endl;')
    driver_code.append("            return 1;")
    driver_code.append("        }")
    driver_code.append("    }")
    driver_code.append("")

    for func_name, bounds, params, return_type in functions:
        for param in params:
            param_tokens = param.strip().split()
            if len(param_tokens) >= 2:
                param_name = param_tokens[-1]
            else:
                exit(f"Cannot parse parameter: {param}")
            try:
                min_val = bounds[param_name]["min"]
                max_val = bounds[param_name]["max"]
            except KeyError:
                exit(
                    f"WARNING: Bounds not found for {param_name} in function {func_name}, manually specify the bounds."
                )
            dist_name = f"{func_name}_{param_name}_dist"
            driver_code.append(f"    std::uniform_real_distribution<{return_type}> {dist_name}({min_val}, {max_val});")
    driver_code.append("")

    driver_code.append("    double sum = 0.;")
    driver_code.append("")

    driver_code.append("    auto start_time = std::chrono::high_resolution_clock::now();")
    driver_code.append("")

    for func_name, bounds, params, return_type in functions:
        driver_code.append(f"    for (int i = 0; i < {num_samples_per_func}; ++i) {{")

        call_params = []
        for param in params:
            param_tokens = param.strip().split()
            if len(param_tokens) >= 2:
                param_name = param_tokens[-1]
            else:
                exit(f"Cannot parse parameter: {param}")
            dist_name = f"{func_name}_{param_name}_dist"
            param_value = f"{dist_name}(gen)"
            param_var_name = f"{param_name}_val"
            driver_code.append(f"        double {param_var_name} = {param_value};")
            call_params.append(param_var_name)

        call_params_str = ", ".join(call_params)

        driver_code.append(f"        double res = {func_name}({call_params_str});")
        driver_code.append("        sum += res;")

        driver_code.append("        if (save_outputs) {")
        driver_code.append(
            '            ofs << std::setprecision(std::numeric_limits<double>::digits10 + 1) << res << "\\n";'
        )
        driver_code.append("        }")

        driver_code.append("    }")
        driver_code.append("")

    driver_code.append('    std::cout << "Sum: " << sum << std::endl;')
    driver_code.append("    auto end_time = std::chrono::high_resolution_clock::now();")
    driver_code.append("    std::chrono::duration<double> elapsed = end_time - start_time;")
    driver_code.append('    std::cout << "Total runtime: " << elapsed.count() << " seconds\\n";')
    driver_code.append("")

    driver_code.append("    if (save_outputs) {")
    driver_code.append("        ofs.close();")
    driver_code.append("    }")
    driver_code.append("")

    driver_code.append("    return 0;")
    driver_code.append("}")
    return "\n".join(driver_code)


def main():
    if len(sys.argv) < 4:
        exit(
            f"Usage: fpopt-golden-driver-generator.py <source_path> <dest_path> <PREC> [func_regex] [num_samples_per_func (default: {DEFAULT_NUM_SAMPLES})]"
        )

    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    PREC = sys.argv[3]
    func_regex = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_REGEX
    num_samples_per_func = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_NUM_SAMPLES

    functions, original_content, matches = parse_c_file(source_path, func_regex)

    driver_code = create_driver_function(functions, num_samples_per_func)

    with open(source_path, "r") as original_file:
        original_content = original_file.read()

    mpfr_header = f'#include "mpfrcpp.hpp"\nconst unsigned int PREC = {PREC};\n#define double mpfrcpp<PREC>\n\n'

    if matches:
        first_match = matches[0]
        insert_pos = first_match.start()
        modified_content = original_content[:insert_pos] + mpfr_header + original_content[insert_pos:]
    else:
        exit("No matching functions found to insert mpfr header.")

    with open(dest_path, "w") as new_file:
        new_file.write(modified_content)
        new_file.write("\n\n" + driver_code)

    print(f"Driver program written to: {dest_path}")


if __name__ == "__main__":
    main()
