#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::cerr << "test failure at line " << line << ": " << expression << '\n';
        exit(1);
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

}  // namespace

int main(int argc, char ** argv) {
    REQUIRE(argc == 2);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "llama-moe-measure-cli-test";
    const std::filesystem::path log = root / "stderr.txt";
    std::filesystem::create_directories(root);

    const std::string command =
        "\"" + std::string(argv[1]) + "\" -m missing.gguf --text missing.txt > \"" +
        log.string() + "\" 2>&1";
    REQUIRE(std::system(command.c_str()) != 0);

    std::ifstream input(log);
    const std::string output((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(output.find("model, output, context size, and parallel slot count are required") != std::string::npos);
    REQUIRE(!std::filesystem::exists(root / "measurement.moem"));

    const std::string help_command =
        "\"" + std::string(argv[1]) + "\" --help > \"" + log.string() + "\" 2>&1";
    REQUIRE(std::system(help_command.c_str()) == 0);
    std::ifstream help_input(log);
    const std::string help_output(
        (std::istreambuf_iterator<char>(help_input)), std::istreambuf_iterator<char>());
    REQUIRE(help_output.find("--no-parse-special") != std::string::npos);

    const std::string list_command =
        "\"" + std::string(argv[1]) + "\" --list-devices > \"" + log.string() + "\" 2>&1";
    REQUIRE(std::system(list_command.c_str()) == 0);
    std::ifstream list_input(log);
    const std::string list_output(
        (std::istreambuf_iterator<char>(list_input)), std::istreambuf_iterator<char>());
    REQUIRE(list_output.find("Available devices:") != std::string::npos);

    std::filesystem::remove_all(root);
    std::cout << "MoE measurement CLI tests passed\n";
    return 0;
}
