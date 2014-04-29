#include <sys/param.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include <sstream>

#include <ailab/common.h>
#include <ext/json.h>

namespace ailab {

/**
 * @brief Tokenizer::Tokenizer Tokenizer which parses simple input which has a fixed set of delimiters
 *        and can be quoted with specific characters
 * @param index
 * @param base
 * @param delims
 * @param quotes
 */
Tokenizer::Tokenizer(int index, std::string& base, std::string delims,
                     std::string quotes)
    : start(index),
      base(base),
      delims(delims),
      quotes(quotes) {
}
Tokenizer::iterator Tokenizer::begin() {
  return iterator(this->base, this->delims, this->quotes, this->start);
}
Tokenizer::iterator Tokenizer::end() {
  return iterator(this->base, this->delims, this->quotes, this->base.size());
}

Tokenizer::iterator::iterator(std::string &base, std::string delims,
                              std::string quotes, size_t i)
    : _base(base),
      _delims(delims),
      _quotes(quotes),
      start(i),
      end(i) {
  // This initializes the "end" position
  ++(*this);
}

const Tokenizer::iterator& Tokenizer::iterator::operator ++() {
  size_t x;
  char c;
  char quoted_with = '\0';
  this->start = std::min(this->_base.size(), this->end);
  for (x = this->start; x < this->_base.size(); x++) {
    c = this->_base[x];
    if (_delims.find(c) != std::string::npos && quoted_with == '\0') {
      break;
    } else if (quoted_with != '\0') {
      if (c == quoted_with)
        quoted_with = '\0';

    } else if (_quotes.find(c) != std::string::npos) {
      quoted_with = c;
    }
  }

  this->end = x + 1;
  this->tok.assign(this->_base.substr(this->start, x - this->start));
  return *this;
}

bool Tokenizer::iterator::operator !=(const Tokenizer::iterator& right) {
  return this->start != right.start;
}

std::string& Tokenizer::iterator::operator *() {
  return this->tok;
}

/**
 * BasicStats
 **/

BasicStats::BasicStats(DVec &raw) {
  this->n = raw.size();
  double fcount = raw.size();
  DVec data(raw.begin(), raw.end());
  std::sort(data.begin(), data.end());
  this->min = data.front();
  this->max = data.back();
  this->median = data.at(data.size() / 2);

  double s = 0;
  for (decimal_t v : data)
    s += v;
  this->mean = s / fcount;

  s = 0;
  for (decimal_t v : data)
    s += ::pow(v - this->mean, 2);
  this->stdev = ::sqrt(s / fcount);
}

std::string BasicStats::toJson() {
  std::stringstream str;
  str << "{";
  str << "n:" << this->n;
  str << ",\"min\":" << this->min;
  str << ",\"max\":" << this->max;
  str << ",\"median\":" << this->median;
  str << ",\"mean\":" << this->mean;
  str << ",\"stdev\":" << this->stdev;
  str << "}";
  return str.str();
}

/**
 *  Logging Framework
 */
std::shared_ptr<LogWriter> (*Logger::factory)(
    std::string key) = &Logger::initSTDOUT;
std::string Logger::location = "";
std::mutex Logger::mtx;
LogMap Logger::inst;

LogWriter::LogWriter()
    : time(std::chrono::steady_clock::now()) {
}

std::string LogWriter::getString(std::stringstream &json) {
  auto delta = std::chrono::steady_clock::now() - this->time;
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
      delta);
  std::stringstream output;
  output << "{ \"usec\":" << microseconds.count() << json.str() << "}\n";
  return output.str();
}

std::shared_ptr<LogWriter> Logger::initFS(std::string key) {
  return std::shared_ptr<LogWriter>(new JSONFSLogger(Logger::location, key));
}

std::shared_ptr<LogWriter> Logger::initHTTP(std::string key) {
  return std::shared_ptr<LogWriter>(new HTTPLogger(Logger::location, key));
}

std::shared_ptr<LogWriter> Logger::initSTDOUT(std::string prefix) {
  return std::shared_ptr<LogWriter>(new STDOUTLogger(prefix));
}

void Logger::log(std::string key, std::stringstream& json) {
  std::unique_lock<std::mutex> lock(Logger::mtx);
  if (Logger::inst.find(key) == Logger::inst.end()) {
    Logger::inst.insert(
        std::pair<std::string, std::shared_ptr<LogWriter> >(
            key, Logger::factory(key)));
  }

  Logger::inst[key]->log(json);
}

JSONFSLogger::JSONFSLogger(std::string filebase, std::string key) {
  std::string filename = filebase + "." + key;
  this->f = std::fopen(filename.c_str(), "a");
  if (this->f == NULL) {
    std::cerr << "Could not open " << filename << std::endl;
  }
}

bool JSONFSLogger::good() {
  return this->f != NULL;
}

JSONFSLogger::~JSONFSLogger() {
  std::fclose(this->f);
}

void JSONFSLogger::log(std::stringstream &json) {
  std::string output = this->getString(json);
  ::fwrite(output.c_str(), output.length(), 1, this->f);
}

bool STDOUTLogger::good() {
  return true;
}

void STDOUTLogger::log(std::stringstream &json) {
  std::cout << this->getString(json);
}

// STDOUT Logging
STDOUTLogger::STDOUTLogger(std::string prefix)
    : prefix(prefix) {
}

// HTTP Logging
HTTPLogger::HTTPLogger(std::string urlbase, std::string key) {
  std::string url = urlbase + "_" + key;

  curl_global_init(CURL_GLOBAL_SSL);
  this->handle = curl_easy_init();
  assert(this->handle != NULL);

  this->_ok = true;
  this->headers = NULL;
  this->headers = curl_slist_append(this->headers,
                                    "Content-Type: application/json");
  this->headers = curl_slist_append(this->headers,
                                    "Transfer-Encoding: chunked");

  curl_easy_setopt(this->handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(this->handle, CURLOPT_POST, 1L);
  curl_easy_setopt(this->handle, CURLOPT_HTTPHEADER, this->headers);

  if (url.find("https://") == 0) {
    curl_easy_setopt(this->handle, CURLOPT_USE_SSL, CURLUSESSL_ALL);
    curl_easy_setopt(this->handle, CURLOPT_SSL_VERIFYPEER, false);
  } else {
    curl_easy_setopt(this->handle, CURLOPT_USE_SSL, CURLUSESSL_NONE);
  }

}

HTTPLogger::~HTTPLogger() {
  curl_slist_free_all(this->headers);
  curl_easy_cleanup(this->handle);
  curl_global_cleanup();
}

bool HTTPLogger::good() {
  return this->_ok;
}

void HTTPLogger::log(std::stringstream &json) {
  std::string output = this->getString(json);
  curl_easy_setopt(this->handle, CURLOPT_POSTFIELDS, output.c_str());
  curl_easy_setopt(this->handle, CURLOPT_POSTFIELDSIZE, output.length());
  this->_ok = (curl_easy_perform(this->handle) == CURLE_OK);
}

// trim from both ends
std::string& trim(std::string &s, std::string bad_chars) {
  size_t first_good = s.find_first_not_of(bad_chars);
  size_t last_good = s.find_last_not_of(bad_chars) + 1;

  if ((last_good != std::string::npos) && (last_good < s.size()))
    s.erase(s.begin() + last_good, s.end());
  if (first_good > 0)
    s.erase(s.begin(), s.begin() + first_good);
  return s;
}

std::string toLower(std::string& s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(), ::tolower);
  return out;
}

std::string toUpper(std::string& s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(), ::toupper);
  return out;
}

std::string abspath_from_relative(std::string target, std::string src,
                                  bool trim_file_from_source) {
  // If target is actually absolute, just leave it alone
  if (target.at(0) == '/')
    return target;

  if (trim_file_from_source) {
    auto p = src.find_last_of('/');
    if (p != std::string::npos)
      src.erase(p + 1, std::string::npos);
  }

  src.append("/");
  src.append(target);

  return src;
}

bool startswith(std::string& base, std::string prefix) {
  if (base.size() < prefix.size())
    return false;
  return base.substr(0, prefix.size()) == prefix;
}

bool endswith(std::string& base, std::string suffix) {
  if (base.size() < suffix.size())
    return false;

  return base.substr(base.size() - suffix.size(), suffix.size()) == suffix;
}

void stringify(std::stringstream& output, json_value *value) {
  if (value->name)
    output << "\"" << value->name << "\" = ";
  switch (value->type) {
    case JSON_NULL:
      output << "null";
      break;
    case JSON_OBJECT:
      output << "{";
      for (json_value *it = value->first_child; it; it = it->next_sibling)
        stringify(output, it);
      output << "}";
      break;
    case JSON_ARRAY:
      output << "[";
      for (json_value *it = value->first_child; it; it = it->next_sibling)
        stringify(output, it);
      output << "]";
      break;
    case JSON_STRING:
      output << "\"" << value->string_value << "\"";
      break;
    case JSON_INT:
      output << value->int_value;
      break;
    case JSON_FLOAT:
      output << value->float_value;
      break;
    case JSON_BOOL:
      output << (value->int_value ? "true" : "false");
      break;
  }
}

std::string stringify(json_value* value) {
  std::stringstream output;
  stringify(output, value);
  return output.str();
}

json_value* parseJSON(char json[], block_allocator* allocator) {
  char *errorPos = 0;
  char *errorDesc = 0;
  int errorLine = 0;
  json_value *root = json_parse(json, &errorPos, &errorDesc, &errorLine,
                                allocator);
  if (root == NULL) {
    std::cerr << "Json Error: [" << errorLine << "] " << errorPos << ": "
              << errorDesc << "\n";
  }
  return root;
}

}
