#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>

class GPT2Tokenizer {
public:
    GPT2Tokenizer();
    ~GPT2Tokenizer() = default;

    // Load standard GPT-2 vocab files
    // vocab_file: maps token string -> integer ID
    // merges_file: text file with "tokenA tokenB" lines
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Main API
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);

private:
    // --- Internal Data Structures ---
    std::map<std::string, int> encoder;       // String -> ID
    std::map<int, std::string> decoder;       // ID -> String
    std::vector<std::pair<std::string, std::string>> bpe_merges; // Priority list of merges
    
    // Cache for BPE processing (memoization)
    // Maps a "word" to its list of BPE tokens
    std::unordered_map<std::string, std::vector<std::string>> cache;

    // The special mapping from Bytes -> Unicode chars (to make them printable)
    std::map<unsigned char, wchar_t> byte_encoder;
    std::map<wchar_t, unsigned char> byte_decoder;

    // --- Helper Functions ---
    void build_byte_encoder();
    std::vector<std::string> bpe(const std::string& token);
    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
};