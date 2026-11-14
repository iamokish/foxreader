#include "../packaging/updater/core.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

std::filesystem::path path_arg(const char* value) {
    std::string decoded, hex=value;
    auto digit=[](char c){return c<='9'?c-'0':c-'a'+10;};
    for (std::size_t i=0;i+1<hex.size();i+=2)
        decoded.push_back(static_cast<char>((digit(hex[i])<<4)|digit(hex[i+1])));
    return std::filesystem::u8path(decoded);
}

int main(int argc, char** argv) {
    if (argc < 5) return 64;
    const std::string mode=argv[1];
    fox_updater::Options options{path_arg(argv[2]),path_arg(argv[3]),argv[4]};
    std::string event=argc>5?argv[5]:"";
    bool triggered=false;
    fox_updater::set_fault_hook([&](const std::string& point) {
        if (point!=event || triggered) return;
        triggered=true;
        if (mode.find("crash")!=std::string::npos) std::_Exit(73);
        throw std::runtime_error("Injected filesystem failure at " + point);
    });
    fox_updater::Result result;
    if (mode.compare(0,7,"recover")==0) result=fox_updater::recover(options);
    else if (mode=="cleanup") result=fox_updater::cleanup(options);
    else if (mode=="inspect") {
        try { std::cout << static_cast<int>(fox_updater::inspect(options)) << "\n"; return 0; }
        catch (const std::exception& error) { std::cout << error.what() << "\n"; return 1; }
    } else {
        result=fox_updater::apply(options,{},[&]{return mode=="cancel";});
    }
    std::cout << result.success << " " << result.cancelled << " " << result.rolled_back
              << " " << result.recovery_required << " " << result.cleanup_pending << "\n"
              << result.message << "\n";
    return 0;
}
