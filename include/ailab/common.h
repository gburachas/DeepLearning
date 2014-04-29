#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <random>
#include <string>
#include <string.h>
#include <memory>
#include <mutex>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

// Might need to update for your system...
#include <curl/curl.h>
#include <ext/json.h>
#include <google/gflags.h>

namespace ailab {

#ifndef decimal_t
    typedef float decimal_t;
#endif

    typedef std::vector<decimal_t> DVec;
    typedef std::shared_ptr<void> ptr;

    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::time_point<Clock> Instant;

    /**
     *  String functions
     **/

    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    class Tokenizer {
    protected:

        std::string& base;
        std::string quotes;
        std::string delims;
        size_t start;
    public:

        class iterator {
        public:
            iterator(std::string& base, std::string _delims, std::string _quotes,
                    size_t i);
            bool operator !=(const iterator& right);
            const iterator& operator++();
            std::string& operator*();
        protected:
            size_t start;
            size_t end;
            std::string _base;
            std::string _quotes;
            std::string _delims;
            std::string tok;
        };

        Tokenizer(int index, std::string& base, std::string delims = ",",
                std::string quotes = "'\"");

        iterator begin();
        iterator end();
    };

    class BasicStats {
    public:
        BasicStats(DVec& raw);
        std::string toJson();
        size_t n;
        double mean;
        double median;
        double max;
        double min;
        double stdev;
    };

    class LogWriter {
    public:

        LogWriter();

        virtual bool good() = 0;
        virtual void log(std::stringstream& json) = 0;

        std::string getString(std::stringstream &json);
        std::chrono::steady_clock::time_point time;
    };

    typedef std::map<std::string, std::shared_ptr<LogWriter> > LogMap;

    class Logger {
    public:
        static std::shared_ptr<LogWriter> initFS(std::string key);
        static std::shared_ptr<LogWriter> initHTTP(std::string key);
        static std::shared_ptr<LogWriter> initSTDOUT(std::string key);

        static void log(std::string key, std::stringstream& json);

        static std::shared_ptr<LogWriter> (*factory)(std::string key);
        static LogMap inst;
        static std::string location;
    protected:

        Logger() {
        }
        Logger(Logger const&);
        void operator=(Logger const&);

        static std::mutex mtx;
    };

    class JSONFSLogger : public LogWriter {
    public:
        JSONFSLogger(std::string filebase, std::string key);
        ~JSONFSLogger();
        bool good();
        void log(std::stringstream &json);
    protected:
        bool at_end;
        FILE* f;
    };

    class STDOUTLogger : public LogWriter {
    public:
        STDOUTLogger(std::string prefix);
        bool good();
        void log(std::stringstream &json);
    protected:
        std::string prefix;
        bool at_end;
    };

    class HTTPLogger : public LogWriter {
    public:
        HTTPLogger(std::string url, std::string key);
        ~HTTPLogger();
        bool good();
        void log(std::stringstream &json);
    protected:
        bool _ok;
        CURL * handle;
        struct curl_slist* headers;
    };

    // trim from both ends
    std::string& trim(std::string &s, std::string bad_chars = " \t\f\v\n\r\'\"");
    std::string toLower(std::string& s);
    std::string toUpper(std::string& s);
    std::string abspath_from_relative(std::string target, std::string src,
            bool trim_file_from_source = false);
    bool startswith(std::string& base, std::string prefix);
    bool endswith(std::string& base, std::string suffix);

    void stringify(std::stringstream& output, json_value *value);
    std::string stringify(json_value* value);
    json_value* parseJSON(char json[], block_allocator* allocator);

    template<typename T>
    void split(const std::string &s, std::vector<T>& elems,
            std::function<T(std::string) > p, char delim = ',',
            char quote = '"') {

        char prev = '\0';
        bool quoted = false;
        size_t start = 0;

        for (size_t i = 0; i < s.length(); i++) {
            char c = s.at(i);
            if (prev != '\\') {

                if (c == quote) {

                    if (quoted) {
                        elems.push_back(p(s.substr(start, i - start)));
                        start = i + 2;
                    } else {
                        start = i + 1;
                    }

                    quoted = !quoted;

                } else if ((c == delim) && !quoted) {
                    elems.push_back(p(s.substr(start, i - start)));
                    start = i + 1;
                }
            }

            prev = c;
        }
    }

    template<typename T>
    void split(const std::string &s, T* dest, size_t n,
            std::function<T(std::string) > p, char delim = ',',
            char quote = '"') {

        char prev = '\0';
        bool quoted = false;
        size_t start = 0;
        size_t on_n = 0;

        for (size_t i = 0; (i < s.length()) && (on_n < n); i++) {
            char c = s.at(i);
            if (prev != '\\') {

                if (c == quote) {

                    if (quoted) {
                        *(dest++) = p(s.substr(start, i - start));
                        on_n++;
                        start = i + 2;
                    } else {
                        start = i + 1;
                    }

                    quoted = !quoted;

                } else if ((c == delim) && !quoted) {
                    *(dest++) = p(s.substr(start, i - start));
                    on_n++;
                    start = i + 1;
                }
            }

            prev = c;
        }
    }

    template<typename T>
    std::vector<T> split(const std::string &s, std::function<T(std::string) > p,
            char delim = ',', char quote = '"') {
        std::vector<T> elems;
        split(s, elems, p, delim, quote);
        return elems;
    }

    template<typename T>
    void splitSparse(const std::string &s
            , T* dest
            , const size_t n // Length of T
            , size_t start
            , size_t end // Because the last character(s) may not work
            , const size_t offset
            , std::function<T(std::string) > p
            , const char delimField // char between fields
            , const char delimValue // char between field id and value
            ) {

        size_t valDelimAt = 0;
        end = std::max(end, s.length());
        while(s.at(start) == delimField){ start++; }
        for (size_t i = start; (i < end && start < end); i++) {

            char c = s.at(i);
            if (c == delimField || i == end) {

                std::string indexStr = s.substr(start, valDelimAt - start);
                auto index = std::atoi(indexStr.c_str()) - offset;

                assert(index < n);
                T value = p(s.substr(valDelimAt + 1, i - valDelimAt));
                dest[ index ] = value;
                start = i + 1;

                while(s.at(start) == delimField){ start++; }

            } else if (c == delimValue) {
                valDelimAt = i;
            }

        }

    }


}
#endif // COMMON_H
