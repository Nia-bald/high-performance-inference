#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include <codecvt>
#include <locale>
#include <algorithm>

// Helper to split string by whitespace/punctuation (Regex mimic)
// GPT-2 regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// For C++, we will use a simplified whitespace splitter for this MVP, 
// but strictly you should use std::regex with the pattern above.
std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> words;
    // Simple space-based split for brevity. 
    // In production, use the full GPT-2 regex logic.
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        // Note: GPT-2 preserves leading spaces as a specific character (Ġ).
        // This is a simplified "Chat" version.
        words.push_back(word);
    }
    return words;
}

GPT2Tokenizer::GPT2Tokenizer() {
    build_byte_encoder();
}

// 1. Build Byte->Unicode Mapping
// GPT-2 maps bytes to printable unicode characters to avoid control chars.
void GPT2Tokenizer::build_byte_encoder() {
    std::vector<unsigned char> bs;
    // Standard ASCII printables
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);
    for (int i = 161; i <= 172; i++) bs.push_back(i);
    for (int i = 174; i <= 255; i++) bs.push_back(i);

    int n = 0;
    for (unsigned char b : bs) {
        byte_encoder[b] = (wchar_t)b; // Identity for printables
        byte_decoder[(wchar_t)b] = b;
    }

    // Map remaining 256 bytes to unique unicode points starting at 256
    for (int b = 0; b < 256; b++) {
        if (byte_encoder.find(b) == byte_encoder.end()) {
            byte_encoder[b] = (wchar_t)(256 + n);
            byte_decoder[(wchar_t)(256 + n)] = b;
            n++;
        }
    }
}

// 2. Load Vocab and Merges
bool GPT2Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // A. Load Encoder from vocab.json
    // The real GPT-2 vocab.json is a single-line JSON: {"token": id, ...}
    std::ifstream v_file(vocab_path);
    if (!v_file.is_open()) {
        std::cerr << "[Tokenizer] Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    // Read entire file into a string
    std::string json_content((std::istreambuf_iterator<char>(v_file)),
                              std::istreambuf_iterator<char>());
    v_file.close();
    
    // Parse JSON: iterate character by character
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < json_content.size() && isspace(json_content[pos])) pos++;
    };
    
    auto parse_string = [&]() -> std::string {
        std::string result;
        pos++; // skip opening quote
        while (pos < json_content.size() && json_content[pos] != '"') {
            if (json_content[pos] == '\\') {
                pos++;
                if (pos >= json_content.size()) break;
                switch (json_content[pos]) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'u': {
                        // Parse 4-hex-digit unicode escape
                        if (pos + 4 < json_content.size()) {
                            std::string hex = json_content.substr(pos + 1, 4);
                            unsigned int codepoint = std::stoul(hex, nullptr, 16);
                            // Convert to UTF-8 (for BMP characters)
                            if (codepoint <= 0x7F) {
                                result += (char)codepoint;
                            } else if (codepoint <= 0x7FF) {
                                result += (char)(0xC0 | (codepoint >> 6));
                                result += (char)(0x80 | (codepoint & 0x3F));
                            } else {
                                result += (char)(0xE0 | (codepoint >> 12));
                                result += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                                result += (char)(0x80 | (codepoint & 0x3F));
                            }
                            pos += 4;
                        }
                        break;
                    }
                    default: result += json_content[pos]; break;
                }
            } else {
                result += json_content[pos];
            }
            pos++;
        }
        pos++; // skip closing quote
        return result;
    };
    
    auto parse_int = [&]() -> int {
        skip_ws();
        size_t start = pos;
        if (pos < json_content.size() && json_content[pos] == '-') pos++;
        while (pos < json_content.size() && isdigit(json_content[pos])) pos++;
        return std::stoi(json_content.substr(start, pos - start));
    };
    
    skip_ws();
    if (pos < json_content.size() && json_content[pos] == '{') pos++; // skip '{'
    
    while (pos < json_content.size()) {
        skip_ws();
        if (json_content[pos] == '}') break;
        if (json_content[pos] == ',') { pos++; continue; }
        
        // Parse "key": value
        if (json_content[pos] == '"') {
            std::string key = parse_string();
            skip_ws();
            if (pos < json_content.size() && json_content[pos] == ':') pos++;
            int val = parse_int();
            encoder[key] = val;
            decoder[val] = key;
        } else {
            pos++; // skip unexpected character
        }
    }
    
    std::cout << "[Tokenizer] Loaded " << encoder.size() << " vocab entries" << std::endl;

    // B. Load Merges
    // Format: "u g" (meaning merge 'u' and 'g' -> 'ug')
    std::ifstream m_file(merges_path);
    if (!m_file.is_open()) return false;

    // Skip first line (version comment)
    std::string line;
    std::getline(m_file, line); 

    while (std::getline(m_file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string p1, p2;
        ss >> p1 >> p2;
        bpe_merges.push_back({p1, p2});
    }

    return true;
}

// 3. Helper: Get set of adjacent pairs in a word list
std::set<std::pair<std::string, std::string>> GPT2Tokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.insert({word[i], word[i+1]});
    }
    return pairs;
}

// 4. Core BPE Algorithm
std::vector<std::string> GPT2Tokenizer::bpe(const std::string& token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    // Initial Split: Character by character
    std::vector<std::string> word;
    // Note: In real implementation, handle multi-byte chars correctly here
    for (char c : token) {
        word.push_back(std::string(1, c));
    }

    // Iteratively merge
    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) break;

        // Find the "best" pair to merge (lowest index in bpe_merges list)
        std::pair<std::string, std::string> best_pair;
        int best_rank = 1e9;
        bool found = false;

        for (const auto& pair : pairs) {
            // Find rank in merges list
            // Linear scan for simplicity; map would be O(1)
            for (int i = 0; i < bpe_merges.size(); ++i) {
                if (bpe_merges[i] == pair) {
                    if (i < best_rank) {
                        best_rank = i;
                        best_pair = pair;
                        found = true;
                    }
                    break;
                }
            }
        }

        if (!found) break; // No more valid merges

        // Perform the merge
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            // If we find the pair, merge it
            if (i < word.size() - 1 && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                new_word.push_back(word[i] + word[i+1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
        if (word.size() == 1) break;
    }

    cache[token] = word;
    return word;
}

// 5. Encode (Text -> IDs)
std::vector<int> GPT2Tokenizer::encode(const std::string& text) {
    std::vector<int> bpe_tokens;
    
    // In reality, map text bytes to special unicode chars first using byte_encoder
    // For simplicity, we assume ASCII input for this C++ demo
    
    auto words = pre_tokenize(text);

    for (const auto& word : words) {
        auto bpe_word = bpe(word);
        for (const auto& token : bpe_word) {
            if (encoder.count(token)) {
                bpe_tokens.push_back(encoder[token]);
            } else {
                // Fallback (shouldn't happen in Byte-BPE if logic is perfect)
                // In Byte-BPE, every byte is in vocab.
                // For this mock, we skip or add UNK
            }
        }
    }
    return bpe_tokens;
}

// 6. Decode (IDs -> Text)
std::string GPT2Tokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids) {
        if (decoder.count(id)) {
            const std::string& token_str = decoder[id];
            // Reverse the byte-to-unicode encoding
            // Each character in token_str is a unicode char from the byte_encoder mapping
            // We need to convert it back to the original byte
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            std::wstring wide = converter.from_bytes(token_str);
            for (wchar_t wc : wide) {
                if (byte_decoder.count(wc)) {
                    text += (char)byte_decoder[wc];
                } else {
                    // Fallback: just convert directly
                    text += converter.to_bytes(wc);
                }
            }
        }
    }
    return text;
}

// 7. Decode a single token ID to string
std::string GPT2Tokenizer::decode_token(int id) {
    return decode({id});
}