#include "core.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <stdio.h>  // renamex_np and RENAME_SWAP keep Finder's bundle path present.
#endif
#endif

namespace fox_updater {
namespace {
namespace fs = std::filesystem;

constexpr std::size_t inventory_limit = 64 * 1024 * 1024;
const std::array<const char*, 14> components = {{
    "bin", "legal", "icons", "launcher", "launcher.exe", "foxreader-updater",
    "foxreader-updater.exe", "libfox-window.so", "libfox-browser.so", "install-desktop.sh",
    "Fox Reader.app", "Fox Reader.command", "build-info.json", "fonts"
}};
constexpr std::size_t fonts_index = 13;
const std::string journal_magic = "FOXREADER-JOURNAL-1\n";

struct Cancelled {};
struct PlanEntry { std::size_t index; bool had_old; bool has_new; };
using Plan = std::vector<PlanEntry>;

#ifdef FOX_UPDATER_TESTING
FaultHook fault_hook;
#endif

void fault(const std::string& event) {
#ifdef FOX_UPDATER_TESTING
    if (fault_hook) fault_hook(event);
#else
    (void)event;
#endif
}

void notify(const ProgressCallback& callback, const std::string& stage,
            const std::string& message, std::uint64_t done = 0,
            std::uint64_t total = 0, bool cancellable = false) noexcept {
    if (callback) {
        try { callback({stage, message, done, total, cancellable}); }
        catch (...) {}  // A display failure cannot interrupt rollback.
    }
}

void check_cancel(const Cancellation& cancel) {
    if (cancel && cancel()) throw Cancelled{};
}

bool present(const fs::path& path) {
    std::error_code error;
    const auto state = fs::symlink_status(path, error);
    if (error && error != std::errc::no_such_file_or_directory &&
        error != std::errc::not_a_directory) throw fs::filesystem_error("Inspect update path", path, error);
    return fs::exists(state);
}

bool reparse(const fs::path& path) {
#ifdef _WIN32
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT);
#else
    (void)path;
    return false;
#endif
}

void plain_directory(const fs::path& path) {
    const auto state = fs::symlink_status(path);
    if (!fs::is_directory(state) || fs::is_symlink(state) || reparse(path))
        throw std::runtime_error("Update directory is missing or redirects elsewhere: " + path.u8string());
}

void regular_file(const fs::path& path) {
    if (!fs::is_regular_file(fs::symlink_status(path)) || reparse(path))
        throw std::runtime_error("Update file is missing or redirects elsewhere: " + path.u8string());
}

bool lower_hex(const std::string& value, std::size_t length) {
    return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

Options paths(const Options& input) {
    Options out = input;
    out.root = fs::canonical(input.root);
    plain_directory(out.root);
    const fs::path transaction = fs::absolute(input.transaction).lexically_normal();
    const std::string name = transaction.filename().u8string();
    const std::string prefix = ".foxreader-update-";
    if (transaction.parent_path() != out.root || name.compare(0, prefix.size(), prefix) != 0 ||
        !lower_hex(name.substr(prefix.size()), 32))
        throw std::runtime_error("Update transaction must be a named direct child of the installation.");
    plain_directory(transaction);
    if (fs::canonical(transaction) != transaction)
        throw std::runtime_error("Update transaction redirects outside the installation.");
    out.transaction = transaction;
    return out;
}

// SHA-256, streamed with a fixed working buffer. Keeping this here avoids a
// dependency on the backend or a platform TLS library in the standalone helper.
class Sha256 {
    std::array<std::uint32_t, 8> state_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<unsigned char, 64> block_{};
    std::uint64_t length_ = 0;
    std::size_t used_ = 0;
    static std::uint32_t rotate(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
        };
        std::uint32_t words[64];
        for (unsigned i = 0; i < 16; ++i)
            words[i] = (std::uint32_t(block_[i*4]) << 24) | (std::uint32_t(block_[i*4+1]) << 16) |
                       (std::uint32_t(block_[i*4+2]) << 8) | block_[i*4+3];
        for (unsigned i = 16; i < 64; ++i) {
            const auto x = words[i-15], y = words[i-2];
            words[i] = words[i-16] + (rotate(x,7)^rotate(x,18)^(x>>3)) + words[i-7] +
                       (rotate(y,17)^rotate(y,19)^(y>>10));
        }
        auto a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        auto e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (unsigned i=0; i<64; ++i) {
            const auto t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+k[i]+words[i];
            const auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }
public:
    void update(const char* bytes, std::size_t size) {
        length_ += size;
        while (size) {
            const auto count = std::min(size, block_.size()-used_);
            std::memcpy(block_.data()+used_, bytes, count);
            used_ += count; bytes += count; size -= count;
            if (used_ == block_.size()) { compress(); used_ = 0; }
        }
    }
    std::string finish() {
        const std::uint64_t bits = length_ * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) { std::fill(block_.begin()+used_, block_.end(), 0); compress(); used_=0; }
        std::fill(block_.begin()+used_, block_.begin()+56, 0);
        for (unsigned i=0; i<8; ++i) block_[63-i] = static_cast<unsigned char>(bits>>(8*i));
        compress();
        std::ostringstream result;
        result << std::hex << std::setfill('0');
        for (auto word : state_) result << std::setw(8) << word;
        return result.str();
    }
};

std::string digest_file(const fs::path& path, const Cancellation& cancel = {}) {
    regular_file(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read update file: " + path.u8string());
    Sha256 digest;
    std::array<char, 128*1024> buffer;
    while (stream) {
        check_cancel(cancel);
        stream.read(buffer.data(), buffer.size());
        digest.update(buffer.data(), static_cast<std::size_t>(stream.gcount()));
    }
    if (!stream.eof()) throw std::runtime_error("Cannot finish reading update file: " + path.u8string());
    return digest.finish();
}

std::string read_small(const fs::path& path, std::size_t limit) {
    regular_file(path);
    const auto size = fs::file_size(path);
    if (size > limit) throw std::runtime_error("Update metadata exceeds its size limit.");
    std::ifstream stream(path, std::ios::binary);
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!stream || !stream.read(result.data(), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("Cannot read update metadata.");
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Update metadata changed while reading.");
    return result;
}

void sync_directory(const fs::path& path) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Open update directory for synchronization");
    const int result = ::fsync(fd), saved = errno;
    ::close(fd);
    if (result) throw std::system_error(saved, std::generic_category(), "Synchronize update directory");
#else
    (void)path;  // Moves use MOVEFILE_WRITE_THROUGH; journal data uses FlushFileBuffers.
#endif
}

void move(const fs::path& source, const fs::path& destination, const std::string& operation,
          bool replace_file = false) {
    if (replace_file) {
        regular_file(source);
        if (present(destination)) regular_file(destination);
    } else if (present(destination)) {
        throw std::runtime_error("Update destination unexpectedly exists: " + destination.u8string());
    }
    fault("before:" + operation);
#ifdef _WIN32
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    const DWORD flags=MOVEFILE_WRITE_THROUGH | (replace_file ? MOVEFILE_REPLACE_EXISTING : 0);
    while (!MoveFileExW(source.c_str(), destination.c_str(), flags)) {
        const DWORD error=GetLastError();
        if ((error!=ERROR_SHARING_VIOLATION && error!=ERROR_LOCK_VIOLATION && error!=ERROR_ACCESS_DENIED) ||
            std::chrono::steady_clock::now()>=deadline)
            throw fs::filesystem_error("Move update component",source,destination,
                                      std::error_code(error,std::system_category()));
        // Backend/launcher shutdown and virus scanners can release handles a
        // moment after process-wait completion. Never force-kill an owner.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    fs::rename(source, destination);
    sync_directory(source.parent_path());
    if (source.parent_path() != destination.parent_path()) sync_directory(destination.parent_path());
#endif
    fault("after:" + operation);
}

#ifdef __APPLE__
void exchange_bundle(const fs::path& source, const fs::path& destination, const std::string& operation) {
    plain_directory(source);
    plain_directory(destination);
    fault("before:" + operation);
    // Do not fall back to two renames: Finder must always have a complete
    // launcher to start recovery, even after an interrupted replacement.
    if (::renamex_np(source.c_str(), destination.c_str(), RENAME_SWAP))
        throw fs::filesystem_error("Exchange application bundles",source,destination,
                                   std::error_code(errno,std::generic_category()));
    sync_directory(source.parent_path());
    if (source.parent_path()!=destination.parent_path()) sync_directory(destination.parent_path());
    fault("after:" + operation);
}
#endif

void durable_write(const fs::path& destination, const std::string& bytes) {
    fs::path temporary = destination;
    temporary += ".tmp";
    if (present(temporary)) fs::remove(temporary);
#ifdef _WIN32
    HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::system_error(GetLastError(), std::system_category(), "Create update journal");
    DWORD written = 0;
    const bool ok = WriteFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) &&
                    written == bytes.size() && FlushFileBuffers(file);
    const DWORD error = GetLastError();
    CloseHandle(file);
    if (!ok) throw std::system_error(error, std::system_category(), "Write update journal");
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(GetLastError(), std::system_category(), "Publish update journal");
#else
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Create update journal");
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = ::write(fd, bytes.data()+offset, bytes.size()-offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { const int error = errno; ::close(fd); throw std::system_error(error, std::generic_category(), "Write update journal"); }
        offset += static_cast<std::size_t>(count);
    }
    const int result = ::fsync(fd), error = errno;
    ::close(fd);
    if (result) throw std::system_error(error, std::generic_category(), "Synchronize update journal");
    fs::rename(temporary, destination);
    sync_directory(destination.parent_path());
#endif
}

std::string unhex(const std::string& value) {
    if (value.empty() || value.size() > 32768 || value.size()%2 || !lower_hex(value, value.size()))
        throw std::runtime_error("Invalid encoded path in update inventory.");
    std::string out;
    out.reserve(value.size()/2);
    auto digit = [](char c) { return c <= '9' ? c-'0' : c-'a'+10; };
    for (std::size_t i=0; i<value.size(); i+=2)
        out.push_back(static_cast<char>((digit(value[i])<<4) | digit(value[i+1])));
    if (std::any_of(out.begin(), out.end(), [](unsigned char c) { return c < 32 || c == 127; }))
        throw std::runtime_error("Control character in update path.");
    return out;
}

fs::path relative_path(const std::string& value) {
    if (value.empty() || value.front() == '/' || value.back() == '/' ||
        value.find('\\') != std::string::npos || value.find(':') != std::string::npos)
        throw std::runtime_error("Invalid update inventory path.");
    const fs::path path = fs::u8path(value);
    if (path.is_absolute() || path.has_root_name()) throw std::runtime_error("Absolute update path rejected.");
    for (const auto& part : path) {
        const std::string text = part.u8string();
        if (text.empty() || text == "." || text == ".." || text.back() == ' ' || text.back() == '.')
            throw std::runtime_error("Unsafe update path component.");
        std::string base = text.substr(0, text.find('.'));
        std::transform(base.begin(), base.end(), base.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
            (base.size() == 4 && (base.compare(0,3,"COM") == 0 || base.compare(0,3,"LPT") == 0) &&
             base[3] >= '1' && base[3] <= '9'))
            throw std::runtime_error("Reserved device name in update path.");
    }
    if (path.generic_u8string() != value) throw std::runtime_error("Noncanonical update path.");
    return path;
}

void validate_link(const fs::path& relative, const std::string& target) {
    if (target.empty() || target.find('\\') != std::string::npos || target.find(':') != std::string::npos)
        throw std::runtime_error("Unsafe update symlink target.");
    const fs::path link = fs::u8path(target);
    if (link.is_absolute() || link.has_root_name()) throw std::runtime_error("Absolute update symlink rejected.");
    const fs::path resolved = (relative.parent_path()/link).lexically_normal();
    if (resolved.empty() || *resolved.begin() == "..") throw std::runtime_error("Update symlink leaves its payload.");
}

struct InventoryEntry { char kind; std::uintmax_t size = 0; std::string value; };

std::map<std::string, InventoryEntry> inventory(const Options& options, const Cancellation& cancel) {
    const fs::path file = options.transaction/"inventory.txt";
    if (!lower_hex(options.inventory_sha256, 64)) throw std::runtime_error("Invalid authenticated inventory digest.");
    const std::string bytes = read_small(file, inventory_limit);
    Sha256 hash; hash.update(bytes.data(), bytes.size());
    if (hash.finish() != options.inventory_sha256) throw std::runtime_error("Update inventory checksum does not match the handoff.");
    std::istringstream input(bytes);
    std::string line;
    if (!std::getline(input, line) || line != "FOXREADER-UPDATER-INVENTORY 1")
        throw std::runtime_error("Unsupported update inventory format.");
    std::map<std::string, InventoryEntry> result;
    std::set<std::string> portable_names;
    while (std::getline(input, line)) {
        check_cancel(cancel);
        std::vector<std::string> parts;
        std::size_t start=0, end;
        while ((end=line.find('\t',start)) != std::string::npos) { parts.push_back(line.substr(start,end-start)); start=end+1; }
        parts.push_back(line.substr(start));
        InventoryEntry entry{};
        std::string name;
        if (parts.size()==2 && parts[0]=="D") { entry.kind='D'; name=unhex(parts[1]); }
        else if (parts.size()==3 && parts[0]=="L") { entry.kind='L'; entry.value=unhex(parts[1]); name=unhex(parts[2]); }
        else if (parts.size()==4 && parts[0]=="F") {
            entry.kind='F'; entry.value=parts[2]; name=unhex(parts[3]);
            if (parts[1].empty() || parts[1].size()>20 || !std::all_of(parts[1].begin(),parts[1].end(),[](char c){ return c>='0'&&c<='9'; }) || !lower_hex(entry.value,64))
                throw std::runtime_error("Invalid update inventory file record.");
            entry.size=std::stoull(parts[1]);
        } else throw std::runtime_error("Invalid update inventory record.");
        const fs::path relative=relative_path(name);
        const std::string top=relative.begin()->u8string();
        if (std::find(components.begin(),components.end(),top)==components.end())
            throw std::runtime_error("Update payload contains a protected or unknown root: " + top);
        if (entry.kind=='L') validate_link(relative,entry.value);
        std::string folded=name;
        std::transform(folded.begin(),folded.end(),folded.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
        if (!portable_names.insert(folded).second || !result.emplace(name,entry).second || result.size()>500000)
            throw std::runtime_error("Duplicate or excessive update inventory entries.");
    }
    if (result.empty() || bytes.empty() || bytes.back()!='\n') throw std::runtime_error("Incomplete update inventory.");
    return result;
}

void verify(const Options& options, const ProgressCallback& callback, const Cancellation& cancel) {
    const fs::path payload=options.transaction/"payload";
    plain_directory(payload);
    const auto expected=inventory(options,cancel);
    std::size_t seen=0;
    notify(callback,"verify","Verifying prepared update files",0,expected.size(),true);
    for (const auto& item : fs::recursive_directory_iterator(payload)) {
        check_cancel(cancel);
        const fs::path relative=item.path().lexically_relative(payload);
        const auto found=expected.find(relative.generic_u8string());
        if (found==expected.end()) throw std::runtime_error("Prepared update contains an unlisted file: " + relative.u8string());
        const auto state=item.symlink_status();
        const auto& entry=found->second;
        if (fs::is_symlink(state)) {
            if (entry.kind!='L' || fs::read_symlink(item.path()).generic_u8string()!=entry.value)
                throw std::runtime_error("Prepared symlink differs from the verified inventory.");
            // A lexical check alone misses links such as a/../../outside when
            // a itself redirects to a shallower directory. Resolve the entire
            // chain in the staged tree, rejecting dangling links and cycles.
            const auto resolved=fs::canonical(item.path());
            const auto inside=resolved.lexically_relative(payload);
            if (inside.empty() || inside.is_absolute() || *inside.begin()=="..")
                throw std::runtime_error("Prepared symlink resolves outside the update payload.");
        } else if (reparse(item.path())) throw std::runtime_error("Unsupported redirection in update payload.");
        else if (fs::is_directory(state)) {
            if (entry.kind!='D') throw std::runtime_error("Prepared directory differs from the verified inventory.");
        } else if (fs::is_regular_file(state)) {
            if (entry.kind!='F' || fs::file_size(item.path())!=entry.size || digest_file(item.path(),cancel)!=entry.value)
                throw std::runtime_error("Prepared file checksum differs from the verified inventory: " + relative.u8string());
        } else throw std::runtime_error("Unsupported file type in update payload.");
        notify(callback,"verify","Verified " + relative.generic_u8string(),++seen,expected.size(),true);
    }
    if (seen!=expected.size()) throw std::runtime_error("Prepared update is missing files from its inventory.");
    plain_directory(payload/"bin");
    plain_directory(payload/"legal");
    regular_file(payload/"build-info.json");
#ifdef _WIN32
    regular_file(payload/"launcher.exe");
#else
    regular_file(payload/"launcher");
#endif
    for (const char* name : {"fonts", "icons", "Fox Reader.app"})
        if (present(payload/name)) plain_directory(payload/name);
}

// Never traverse a symlink, junction or other reparse point while removing
// staged/old trees. Every call originates inside a validated transaction.
void remove_tree(const fs::path& path) {
    const auto state=fs::symlink_status(path);
    if (!fs::exists(state)) return;
    if (!fs::is_symlink(state) && !reparse(path)) {
        if (fs::is_directory(state)) {
            fs::permissions(path,fs::perms::owner_write|fs::perms::owner_exec,fs::perm_options::add);
            for (const auto& entry : fs::directory_iterator(path)) remove_tree(entry.path());
        } else fs::permissions(path,fs::perms::owner_write,fs::perm_options::add);
    }
    fs::remove(path);
}

void durable_copy(const fs::path& source, const fs::path& destination, const Cancellation& cancel,
                  const std::string& operation = {}) {
    std::ifstream input(source,std::ios::binary);
    std::ofstream output(destination,std::ios::binary|std::ios::trunc);
    if (!input || !output) throw std::runtime_error("Cannot copy a file into the update transaction.");
    std::array<char,128*1024> buffer;
    while (input) {
        check_cancel(cancel);
        input.read(buffer.data(),buffer.size());
        output.write(buffer.data(),input.gcount());
        if (!output) throw std::runtime_error("Writing an update copy failed; check free disk space.");
        if (!operation.empty()) fault("during:" + operation);
    }
    if (!input.eof()) throw std::runtime_error("Reading an existing file failed.");
    output.flush();
    if (!output) throw std::runtime_error("Finishing an update copy failed.");
    output.close();
#ifdef _WIN32
    HANDLE file=CreateFileW(destination.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,nullptr);
    if (file==INVALID_HANDLE_VALUE)
        throw std::system_error(GetLastError(),std::system_category(),"Open update copy for synchronization");
    const bool ok=FlushFileBuffers(file)!=FALSE;
    const DWORD error=GetLastError();
    CloseHandle(file);
    if (!ok) throw std::system_error(error,std::system_category(),"Synchronize update copy");
    fs::permissions(destination,fs::symlink_status(source).permissions());
#else
    // Publish executable permissions durably with the launcher bytes.
    fs::permissions(destination,fs::symlink_status(source).permissions());
    const int fd=::open(destination.c_str(),O_RDONLY);
    if (fd<0) throw std::system_error(errno,std::generic_category(),"Open update copy for synchronization");
    const int result=::fsync(fd), error=errno;
    ::close(fd);
    if (result) throw std::system_error(error,std::generic_category(),"Synchronize update copy");
#endif
}

void copy_overlay(const fs::path& source, const fs::path& destination, const Cancellation& cancel,
                  const std::string& operation = {}) {
    check_cancel(cancel);
    const auto state=fs::symlink_status(source);
    if (fs::is_symlink(state)) {
        if (present(destination)) remove_tree(destination);
        fs::copy_symlink(source,destination);
    } else if (reparse(source)) throw std::runtime_error("Update copy contains an unsupported redirection.");
    else if (fs::is_directory(state)) {
        if (present(destination) && (!fs::is_directory(fs::symlink_status(destination)) || reparse(destination)))
            remove_tree(destination);
        if (!present(destination)) fs::create_directory(destination);
        fs::permissions(destination,fs::perms::owner_write|fs::perms::owner_exec,fs::perm_options::add);
        for (const auto& entry : fs::directory_iterator(source))
            copy_overlay(entry.path(),destination/entry.path().filename(),cancel,operation);
        fs::permissions(destination,state.permissions());
        sync_directory(destination);
    } else if (fs::is_regular_file(state)) {
        if (present(destination)) remove_tree(destination);
        durable_copy(source,destination,cancel,operation);
    } else throw std::runtime_error("Unsupported file in an update copy.");
}

bool launch_file(const PlanEntry& entry) {
    return entry.index==3 || entry.index==4 || entry.index==11;
}

bool launch_bundle(const PlanEntry& entry) {
#ifdef __APPLE__
    return entry.index==10 && entry.had_old && entry.has_new;
#else
    (void)entry;
    return false;
#endif
}

void backup_launch_entry(const fs::path& source, const fs::path& backup, const std::string& name) {
    fs::path temporary=backup;
    temporary += ".tmp";
    if (present(temporary) || present(backup))
        throw std::runtime_error("Update contains a stale launcher backup.");
    fault("before:copy:"+name);
    copy_overlay(source,temporary,{},"backup:"+name);
    fault("after:copy:"+name);
    // Partial copies never become recovery sources. The live launcher stays
    // in place until its complete durable backup has been atomically published.
    move(temporary,backup,"backup:"+name);
}

fs::path new_path(const Options& options, const PlanEntry& entry) {
    return entry.index==fonts_index ? options.transaction/"fonts-merged" : options.transaction/"payload"/components[entry.index];
}

void save_plan(const Options& options, const Plan& plan) {
    std::string bytes=journal_magic;
    bytes.push_back(static_cast<char>(plan.size()));
    for (const auto& entry : plan) {
        bytes.push_back(static_cast<char>(entry.index));
        bytes.push_back(static_cast<char>((entry.had_old?1:0)|(entry.has_new?2:0)));
    }
    durable_write(options.transaction/"journal.bin",bytes);
    fault("journal");
}

Plan read_plan(const Options& options) {
    const auto bytes=read_small(options.transaction/"journal.bin",1024);
    if (bytes.compare(0,journal_magic.size(),journal_magic)!=0 || bytes.size()<=journal_magic.size())
        throw std::runtime_error("Update recovery journal is invalid.");
    const auto count=static_cast<unsigned char>(bytes[journal_magic.size()]);
    if (!count || count>components.size() || bytes.size()!=journal_magic.size()+1+count*2)
        throw std::runtime_error("Update recovery journal is incomplete.");
    Plan plan;
    std::set<std::size_t> seen;
    for (std::size_t i=0; i<count; ++i) {
        const auto index=static_cast<unsigned char>(bytes[journal_magic.size()+1+i*2]);
        const auto flags=static_cast<unsigned char>(bytes[journal_magic.size()+2+i*2]);
        if (index>=components.size() || !flags || flags>3 || !seen.insert(index).second)
            throw std::runtime_error("Update recovery journal contains invalid components.");
        plan.push_back({index,(flags&1)!=0,(flags&2)!=0});
    }
    return plan;
}

TransactionState state(const Options& options) {
    const auto marker=options.transaction/"state";
    if (present(marker)) {
        const auto value=read_small(marker,32);
        if (value=="COMMITTED\n") return TransactionState::committed;
        if (value=="ROLLED_BACK\n") return TransactionState::rolled_back;
        throw std::runtime_error("Invalid update transaction state.");
    }
    return present(options.transaction/"journal.bin") ? TransactionState::interrupted : TransactionState::prepared;
}

Result rollback(const Options& options, const ProgressCallback& callback) {
    const Plan plan=read_plan(options);
    plain_directory(options.transaction/"old");
    plain_directory(options.transaction/"discard");
    plain_directory(options.transaction/"payload");
    notify(callback,"rollback","Restoring the previous Fox Reader installation",0,plan.size(),false);
    std::size_t done=0;
    for (auto it=plan.rbegin(); it!=plan.rend(); ++it) {
        const auto& entry=*it;
        const std::string name=components[entry.index];
        const auto original=options.root/name, backup=options.transaction/"old"/name;
        const auto discarded=options.transaction/"discard"/name;
        if (entry.had_old) {
            if (present(backup)) {
                if (launch_file(entry) && entry.has_new) {
                    // Replacing a regular launcher consumes the complete backup
                    // atomically. A retry sees either that backup or the already
                    // restored original; it never observes an absent launcher.
                    move(backup,original,"restore:"+name,true);
#ifdef __APPLE__
                } else if (launch_bundle(entry)) {
                    // Keep this backup immutable: after a directory exchange
                    // the temporary path contains the failed new bundle. A
                    // repeated recovery always starts from the old backup.
                    const auto temporary=options.transaction/"restore-app";
                    if (present(temporary)) remove_tree(temporary);
                    fault("before:restore-copy:"+name);
                    copy_overlay(backup,temporary,{},"restore-copy:"+name);
                    sync_directory(options.transaction);
                    fault("after:restore-copy:"+name);
                    if (present(original)) exchange_bundle(temporary,original,"restore:"+name);
                    else move(temporary,original,"restore:"+name);
#endif
                } else {
                    if (present(original)) move(original,discarded,"discard:"+name);
                    move(backup,original,"restore:"+name);
                }
            } else if (!present(original)) {
                throw std::runtime_error("Cannot recover missing original component: " + name);
            }
        } else if (entry.has_new && !present(new_path(options,entry)) && present(original)) {
            move(original,discarded,"remove-new:"+name);
        }
        notify(callback,"rollback","Restored " + name,++done,plan.size(),false);
    }
    durable_write(options.transaction/"state","ROLLED_BACK\n");
    fault("rolled-back");
    Result result; result.rolled_back=true; result.cleanup_pending=true;
    result.message="The previous Fox Reader installation was restored.";
    return result;
}
}  // namespace

#ifdef FOX_UPDATER_TESTING
void set_fault_hook(FaultHook callback) { fault_hook=std::move(callback); }
#endif

TransactionState inspect(const Options& options) { return state(paths(options)); }

Result recover(const Options& input, ProgressCallback callback) {
    Result result;
    try {
        const auto options=paths(input);
        const auto current=state(options);
        if (current==TransactionState::committed) {
            result.success=true; result.cleanup_pending=true; result.message="The update was already installed.";
        } else if (current==TransactionState::rolled_back) {
            result.rolled_back=true; result.cleanup_pending=true; result.message="The previous installation was already restored.";
        } else if (current==TransactionState::prepared) {
            result.cancelled=true; result.cleanup_pending=true; result.message="The installation has not been changed.";
        } else result=rollback(options,callback);
    } catch (const std::exception& error) {
        result.recovery_required=true;
        result.message="Recovery could not finish. Keep the update transaction and retry recovery. " + std::string(error.what());
    }
    return result;
}

Result apply(const Options& input, ProgressCallback callback, Cancellation cancel) {
    Result result;
    Options options;
    bool validated=false;
    try {
        options=paths(input); validated=true;
        if (state(options)!=TransactionState::prepared) return recover(options,callback);
        verify(options,callback,cancel);
        for (const char* name : {"old","discard","fonts-merged"})
            if (present(options.transaction/name)) throw std::runtime_error("Update transaction contains stale working files; prepare a new update.");
        const auto new_fonts=options.transaction/"payload"/"fonts";
        if (present(new_fonts)) {
            notify(callback,"prepare","Preserving your fonts and adding packaged fonts",0,0,true);
            const auto merged=options.transaction/"fonts-merged";
            fs::create_directory(merged);
            if (present(options.root/"fonts")) {
                plain_directory(options.root/"fonts");
                copy_overlay(options.root/"fonts",merged,cancel);
            }
            copy_overlay(new_fonts,merged,cancel);
        }
        Plan plan;
        for (std::size_t i=0; i<components.size(); ++i) {
            const bool has_new=present(options.transaction/"payload"/components[i]);
            const bool had_old=present(options.root/components[i]);
            if (i==fonts_index && !has_new) continue;
            const PlanEntry entry{i,had_old,has_new};
            if (had_old && has_new && launch_file(entry)) {
                regular_file(options.root/components[i]);
                regular_file(options.transaction/"payload"/components[i]);
            }
            if (launch_bundle(entry)) plain_directory(options.root/components[i]);
            if (has_new || had_old) plan.push_back(entry);
        }
        fs::create_directory(options.transaction/"old");
        fs::create_directory(options.transaction/"discard");
#ifdef __APPLE__
        if (std::any_of(plan.begin(),plan.end(),launch_bundle)) {
            // Some portable drives do not support atomic directory exchange.
            // Discover that before changing any installed component, since
            // rollback must use the same primitive to keep Finder usable.
            const auto first=options.transaction/"exchange-probe-a";
            const auto second=options.transaction/"exchange-probe-b";
            if (present(first) || present(second))
                throw std::runtime_error("Update transaction contains a stale exchange probe.");
            fs::create_directory(first);
            fs::create_directory(second);
            exchange_bundle(first,second,"check:bundle-exchange");
            fs::remove(first);
            fs::remove(second);
        }
#endif
        sync_directory(options.transaction);
        sync_directory(options.root);
        check_cancel(cancel);
        save_plan(options,plan);
        notify(callback,"install","Installing Fox Reader; please keep the updater open",0,plan.size(),false);
        std::size_t done=0;
        for (const auto& entry : plan) {
            const std::string name=components[entry.index];
            const auto destination=options.root/name;
            const bool atomic_launch=entry.had_old && entry.has_new && (launch_file(entry) || launch_bundle(entry));
            if (atomic_launch) backup_launch_entry(destination,options.transaction/"old"/name,name);
            else if (entry.had_old) move(destination,options.transaction/"old"/name,"backup:"+name);
            else if (present(destination)) throw std::runtime_error("Installation changed during the update: " + name);
            if (entry.has_new) {
#ifdef __APPLE__
                if (launch_bundle(entry)) exchange_bundle(new_path(options,entry),destination,"install:"+name);
                else
#endif
                move(new_path(options,entry),destination,"install:"+name,atomic_launch);
            }
            notify(callback,"install","Installed " + name,++done,plan.size(),false);
        }
        durable_write(options.transaction/"state","COMMITTED\n");
        fault("committed");
        result.success=true; result.cleanup_pending=true;
        result.message="Fox Reader was updated successfully. You can close this updater and launch Fox Reader when ready.";
        notify(callback,"success",result.message,plan.size(),plan.size(),false);
    } catch (const Cancelled&) {
        result.cancelled=true; result.cleanup_pending=true; result.message="Update cancelled. Your installation has not been changed.";
    } catch (const std::exception& error) {
        const std::string reason=error.what();
        try {
            if (validated && state(options)==TransactionState::interrupted) {
                result=recover(options,callback);
                result.message="Update failed: " + reason + " " + result.message;
            } else if (validated && state(options)==TransactionState::committed) {
                result.success=true; result.cleanup_pending=true; result.message="The update was installed; cleanup is pending.";
            } else {
                result.cleanup_pending=validated;
                result.message="Update could not start. Your installation has not been changed. " + reason;
            }
        } catch (const std::exception& secondary) {
            result.recovery_required=true;
            result.message="Update recovery is required. " + reason + " " + secondary.what();
        }
    }
    return result;
}

Result cleanup(const Options& input) {
    Result result;
    try {
        const auto options=paths(input);
        const auto current=state(options);
        if (current==TransactionState::interrupted) {
            result.recovery_required=true; result.message="An interrupted update must be recovered before cleanup."; return result;
        }
        if (current==TransactionState::prepared) {
            // An unchanged installation is also safe to launch. Publish that
            // terminal state before removing its preparation files, so a
            // cancelled updater killed while Windows still locks its runner
            // can have the remaining cleanup retried by the next launcher.
            durable_write(options.transaction/"state","ROLLED_BACK\n");
        }
        std::vector<fs::path> pending;
        for (const auto& item : fs::directory_iterator(options.transaction)) {
            const auto name=item.path().filename();
            if (name!="journal.bin" && name!="state" && name!="runner") pending.push_back(item.path());
        }
        std::string failures;
        for (const auto& path : pending) {
            try { remove_tree(path); }
            catch (const std::exception& error) { failures=error.what(); }
        }
        try { if (present(options.transaction/"runner")) remove_tree(options.transaction/"runner"); }
        catch (const std::exception& error) { failures=error.what(); }
        if (!failures.empty()) {
            result.cleanup_pending=true; result.message="Update cleanup will finish after the updater closes. " + failures; return result;
        }
        // Journal first: a crash between marker deletions must never make a
        // committed installation look like an interrupted replacement.
        if (present(options.transaction/"journal.bin")) fs::remove(options.transaction/"journal.bin");
        sync_directory(options.transaction);
        if (present(options.transaction/"state")) fs::remove(options.transaction/"state");
        fs::remove(options.transaction);
        sync_directory(options.root);
        result.success=true; result.message="Update files were removed.";
    } catch (const std::exception& error) {
        result.cleanup_pending=true; result.message="Update files could not be fully removed. " + std::string(error.what());
    }
    return result;
}

}  // namespace fox_updater
