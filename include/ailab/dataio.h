#ifndef DATAIO_H
#define DATAIO_H

#include <string>
#include <string.h>
#include <chrono>
#include <cmath>
#include <atomic>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <map>
#include <thread>
#include <condition_variable>
#include <functional>

#include <stdint.h>
#include <ailab/oclmatrix.h>
#include <ailab/common.h>
#include <ailab/exception.h>
#include <ailab/zbin.h>

#define VALFLOOR 1.0e-10f

namespace ailab {

    template<typename T>
    class DataIO {
    public:
        class InputBuffer;
        class OutputBuffer;

        typedef std::shared_ptr<std::ostream> oStream;
        typedef std::shared_ptr<std::istream> iStream;

        typedef std::shared_ptr<InputBuffer> spInput;
        typedef std::shared_ptr<OutputBuffer> spOutput;

        typedef std::function<bool(OCLMatrix<T>&) > Filler;
        typedef std::function<void(OCLMatrix<T>&) > Drainer;
        typedef std::function<void(std::istream&, InputBuffer& buf) > Resetter;

        typedef std::unique_lock<std::mutex> LockT;

        typedef struct {
            std::string protocol;
            std::string path;
            std::string format;
            bool binary;
        } KeyParts;

        class CyclicBuffer {
        public:

            CyclicBuffer(size_t max_rows, bool is_binary)
            : max_rows(max_rows),
            is_binary(is_binary),
            width(0),
            push_at(0),
            pull_from(0),
            num_rows_in(0),
            num_rows_out(0),
            byteWidth(0),
            data(NULL),
            fullyBuffered(false),
            badStream(false),
            has_been_initialized(false) {}

            ~CyclicBuffer() {
                if (this->data != NULL) {
                    free(this->data);
                }
            }

            void set_width(size_t width) {
                if (this->data == NULL) {
                    this->width = width;
                    this->byteWidth = sizeof (T) * width;
                    this->has_been_initialized = true;
                    this->data = (T*) malloc(sizeof (T) * width * this->max_rows);
                    memset(this->data, 0, sizeof (T) * width * this->max_rows);
                } else {
                    assert(this->width == width);
                }
            }

            bool is_ok(size_t low, size_t high) {

                if (low < high) {

                    size_t l_wrap = low / this->max_rows;
                    size_t h_wrap = high / this->max_rows;

                    if (h_wrap > l_wrap) {
                        size_t l_row = low % this->max_rows;
                        size_t h_row = high % this->max_rows;
                        return l_row > h_row;
                    } else {
                        return true;
                    }

                } else {
                    return false;
                }
            }


            /**
             * @brief couldFill
             * @return Check if there might be room for a row
             */
            bool couldFill() {
                return this->is_ok(this->pull_from, this->push_at + 1);
            }

            /**
             * @brief couldDrain
             * @return Check if there might be a row available
             */
            bool couldDrain() {
                return this->is_ok(this->pull_from + 1, this->push_at);
            }

            /**
             * @brief putRow
             * @return Push a row into the buffer
             */
            void putRow(T* src, size_t width) {

                LockT lock(this->mutex);

                if( ! this->couldFill() ){
                    this->countersChanged.wait(lock, [this] {
                        return this->couldFill();
                    });
                }

                size_t row_index = (this->push_at++) % this->max_rows;

                // Move the pointers
                this->num_rows_in++;
                std::copy(src, src + width, this->data + (this->width * row_index));
                memcpy(this->data + (this->width * row_index), src, this->byteWidth);
                this->countersChanged.notify_all();
            }

            /**
             * @brief getRow
             * @return Get a row which can be drained
             */
            bool getRow(T* dest, int length=0) {

                LockT lock(this->mutex);

                if(!this->fullyBuffered)
                {
                    if( !this->couldDrain() ) {
                        this->countersChanged.wait(lock, [this] {
                            return this->couldDrain() || this->badStream;
                        });
                    }

                    if(this->badStream){
                        return false;
                    }
                }

                size_t row_index = (this->pull_from++) % this->max_rows;
                T* srcStart = this->data + (this->width * row_index);
                T* srcEnd = this->data + (this->width * (row_index + 1));

                // Move the pointers
                bool cycled = false;

                if((length > 1) && (this->pull_from > 0))
                {
                    cycled = (this->pull_from % length == 0);
                }

                this->num_rows_out++;
                std::copy(srcStart, srcEnd, dest);
                this->countersChanged.notify_all();
                return cycled;
            }

            const bool is_binary;
            bool has_been_initialized;
            std::atomic_bool badStream;
            std::atomic_bool fullyBuffered;
            size_t max_rows;
            size_t width;
            size_t byteWidth;
            size_t push_at;
            size_t pull_from;

            size_t num_rows_in;
            size_t num_rows_out;

            T* data;

            std::mutex mutex;
            std::condition_variable countersChanged;
        };

        class InputBuffer : public CyclicBuffer {
        public:

            InputBuffer(iStream stream, bool seekable, bool is_binary, size_t max_rows,
                    Resetter seekData)
            : CyclicBuffer(max_rows, is_binary),
            inputStream(stream),
            seekable(seekable),
            totalInputBytes(0),
            total_length(0),
            sourceWidth(0),
            hitEnd(false),
            cycled(false),
            seekData(seekData) {

                if (seekable) {
                    stream->seekg(0, stream->end);
                    this->totalInputBytes = stream->tellg();
                    stream->seekg(0, stream->beg);
                }

            }

            /**
             * @brief fillRows This will greedily fill all possible rows
             */
            virtual void fillRows() {

                T row[ this->width ];

                while (this->inputStream->good() && this->couldFill() ) {
                    // Get the row from the input
                    this->getRowFromStream(row);

                    // Now put it into this buffer
                    this->putRow(row, this->width);
                }

                this->badStream = !this->inputStream->good();
            }

            int onCycle(){
                return total_length == -1 ? 0 : this->pull_from / this->total_length;
            }

            virtual void getRowFromStream(T* row) = 0;

            std::atomic_bool cycled;
            Resetter seekData;
            const bool seekable;
            size_t totalInputBytes;
            unsigned char sourceWidth;
            iStream inputStream;

            std::atomic_bool hitEnd;
            long long total_length;
        };

        class OutputBuffer : public CyclicBuffer {
        public:

            OutputBuffer(oStream outputStream, size_t max_rows, bool binary)
                : CyclicBuffer(max_rows, binary)
                  , outputStream(outputStream){}

            void drain() {
                T row[ this->width ];

                // Get the row from the buffer
                while(this->couldDrain() && this->outputStream->good() && !this->getRow(row, 0))
                {
                    // Write the row to the Output Stream
                    this->pushRowToStream( row );
                }

                this->badStream = !this->outputStream->good();
            }

            virtual void pushRowToStream( T* row ) = 0;

            oStream outputStream;
        };

        class CSVOutputBuffer : public OutputBuffer {
        public:
            CSVOutputBuffer(oStream outputStream, size_t max_rows)
                : OutputBuffer(outputStream, max_rows, false){
                std::cerr << "CSV output buffer" << std::endl;
            }

            void pushRowToStream(T *row){
                std::stringstream ss;
                ss << row[0];
                for (size_t i = 1; i < this->width; i++) {
                    ss << "," << row[i];
                }
                ss << "\n";

                this->outputStream->write(ss.str().c_str(), ss.str().length());
            }

        };

        class SparseArffOutputBuffer : public OutputBuffer {
        public:
            SparseArffOutputBuffer(oStream outputStream, size_t max_rows)
                : OutputBuffer(outputStream, max_rows, false){
                std::cerr << "ARFF output buffer" << std::endl;
            }

            void pushRowToStream(T *row){
                std::stringstream ss;
                size_t i=0;
                ss << '{';
                // Get the first one out
                for (; i < this->width; i++) {
                    if(row[i] != 0){
                        ss << i << ' ' << row[i];
                        break;
                    }
                }

                // Now put a comma before the others
                for (; i < this->width; i++) {
                    if(row[i] != 0){
                        ss << ',' << i << ' ' << row[i];
                        break;
                    }
                }
                ss << "}\n";

                this->outputStream->write(ss.str().c_str(), ss.str().length());
            }
        };

        class SVMOutputBuffer : public OutputBuffer {
        public:
            SVMOutputBuffer(oStream outputStream, size_t max_rows)
                : OutputBuffer(outputStream, max_rows, false){
                std::cerr << "SVM output buffer" << std::endl;
            }

            void pushRowToStream(T *row){
                std::stringstream ss;
                size_t i=0;

                // Get the first one out
                for (; i < this->width; i++) {
                    if(row[i] != 0){
                        ss << i << ':' << row[i];
                        break;
                    }
                }

                // Now put a comma before the others
                for (; i < this->width; i++) {
                    if(row[i] != 0){
                        ss << ' ' << i << ':' << row[i];
                        break;
                    }
                }
                ss << '\n';

                this->outputStream->write(ss.str().c_str(), ss.str().length());
            }

        };

        class ZBinOutputBuffer : public OutputBuffer {
        public:
            ZBinOutputBuffer(oStream outputStream, size_t max_rows)
                : OutputBuffer(outputStream, max_rows, true){
                std::cerr << "ZBIN output buffer" << std::endl;
            }

            void pushRowToStream( T* row ){
                zbin::RowHeader rowHeader;
                rowHeader.maxWidth = this->width;
                rowHeader.realWidth = zbin::denseWidth;
                this->outputStream->write((char*) &rowHeader, sizeof (rowHeader));
                this->outputStream->write((char*) row, this->byteWidth);
            }
        };

        class StringInputBuffer : public InputBuffer {
        public:

            StringInputBuffer(iStream stream, bool seekable, char delim,
                    size_t max_rows, Resetter seekData)
            : InputBuffer(stream, seekable, false, max_rows, seekData),
            delim(delim) {

                std::vector<T> vec;
                std::string line;
                std::getline(*this->inputStream, line);

                split<T>(line, vec, [](std::string str)->T {
                    return ::atof(str.c_str());
                }, this->delim);

                this->set_width(vec.size());
                memcpy(this->data, &vec[0], this->byteWidth);
                this->push_at++;

                this->total_length = -1;
            }

            void getRowFromStream(T* row) {
                std::string line;
                std::getline(*this->inputStream, line);

                while (line.length() > 0 && line.at(0) == '%') {
                    line.assign("");
                    std::getline(*this->inputStream, line);
                }
                if (line.length() > 0) {
                    split<T>(line, row, this->width, [](std::string str)->T {
                        T v = ::atof(str.c_str());
                        return (std::isfinite(v) && std::abs(v) > VALFLOOR) ? v : 0;
                    },
                    this->delim);
                }
            }

            char delim;
        };

        class SparseArffInputBuffer : public InputBuffer {
        public:

            SparseArffInputBuffer(iStream stream, bool seekable,
                    size_t max_rows, Resetter seekData)
            : InputBuffer(stream, seekable, false, max_rows, seekData) {
                this->total_length = -1;
            }

            void getRowFromStream(T* row) {

                // Since this is a sparse format we must zero out the row first
                memset(row, 0, this->byteWidth);
                auto parser = [](std::string str)->T {
                    T v = ::atof(str.c_str());
                    return (std::isfinite(v) && std::abs(v) > VALFLOOR) ? v : 0;
                };

                std::string line;
                std::getline(*this->inputStream, line);

                while (line.length() > 0 && line.at(0) == '%') {
                    line.assign("");
                    std::getline(*this->inputStream, line);
                }

                if (line.length() > 0) {
                    splitSparse<T>(line
                            , row
                            , this->width
                            , 1
                            , line.length() - 1
                            , 0
                            , parser
                            , ','
                            , ' ');
                }
            }

        };

        class SVMLightInputBuffer : public InputBuffer {
        public:

            SVMLightInputBuffer(iStream stream, bool seekable,
                    size_t max_rows, Resetter seekData)
            : InputBuffer(stream, seekable, false, max_rows, seekData) {
                this->total_length = -1;
            }

            void getRowFromStream(T* row) {

                // Since this is a sparse format we must zero out the row first
                memset(row, 0, this->byteWidth);

                auto parser = [](std::string str)->T {
                    T v = ::atof(str.c_str());
                    return (std::isfinite(v) && std::abs(v) > VALFLOOR) ? v : 0;
                };

                std::string line;
                std::getline(*this->inputStream, line);

                while (line.length() > 0 && line.at(0) == '#') {
                    line.assign("");
                    std::getline(*this->inputStream, line);
                }

                if (line.length() > 0) {
                    size_t start = line.find(' ');
                    size_t firstField = line.find(':');

                    if (line.at(start + 1) == 'q') {
                        start = line.find(' ', start + 1);
                    }

                    if (firstField < start) {
                        start = 0;
                    }
                    splitSparse<T>(line, row, this->width, start, line.length(), 1, parser, ' ', ':');
                }
            }

        };

        class BinaryInputBuffer : public InputBuffer {
        public:

            BinaryInputBuffer(iStream stream, bool seekable, size_t max_rows,
                    Resetter seekData)
            : InputBuffer(stream, seekable, true, max_rows, seekData) {

                zbin::RowHeader rh;
                stream->read((char*) &rh, sizeof (zbin::RowHeader));
                this->set_width(rh.maxWidth);
                this->rawRow = (zbin::Record *) malloc(
                        rh.maxWidth * sizeof (zbin::Record));

                if (rh.realWidth == zbin::denseWidth) {
                    stream->read((char*) this->data, rh.maxWidth * sizeof (T));
                } else {
                    memset((void*) this->data, 0, rh.maxWidth * sizeof (T));
                    stream->read((char*) this->rawRow, rh.realWidth * sizeof (zbin::Record));

                    for (size_t i = 0; i < rh.realWidth; i++) {
                        auto v = this->rawRow[i].value;
                        this->data[this->rawRow[i].index] = (std::isfinite(v) && std::abs(v) > VALFLOOR) ? v : 0;
                    }
                }

                this->push_at++;
            }

            virtual ~BinaryInputBuffer() {
                if (this->rawRow != NULL) {
                    free(this->rawRow);
                }
            }

            void getRowFromStream(T* row) {

                zbin::RowHeader rh = {0, 0};
                this->inputStream->read((char*) &rh, sizeof (zbin::RowHeader));
                if (rh.realWidth > 0) {

                    if (this->width != rh.maxWidth) {
                        throw new Exception::BadBinaryData();
                    }

                    if (rh.realWidth == zbin::denseWidth) {
                        this->inputStream->read((char*) row, rh.maxWidth * sizeof (T));

                        for (zbin::width_t i = 0; i < rh.maxWidth; i++) {
                            T v = row[i];
                            if (!(std::isfinite(v) && std::abs(v) > VALFLOOR)) {
                                row[i] = 0;
                            }
                        }

                    } else {

                        this->inputStream->read((char*) this->rawRow,
                                rh.realWidth * sizeof (zbin::Record));

                        memset((void*) row, 0, rh.maxWidth * sizeof (T));
                        for (size_t i = 0; i < rh.realWidth; i++) {
                            T v = this->rawRow[i].value;
                            row[this->rawRow[i].index] =
                                    (std::isfinite(v) && std::abs(v) > VALFLOOR) ? v : 0;
                        }
                    }

                }

            }

            zbin::Record * rawRow;

        };

        void run_thread() {

            while (this->running) {

                {
                    LockT lock(this->mInputs, std::defer_lock);

                    // Try to do an input
                    if ( lock.try_lock() ) {

                        for (auto p : this->inputs) {
                            InputBuffer& buf = *p.second;

                            if (!buf.fullyBuffered) {

                                    if ( buf.inputStream->good() ) {
                                        buf.fillRows();
                                    } else {
                                        LockT lock(buf.mutex);

                                        // At the end we will have space
                                        // so the entire dataset is in the buffer
                                        if (buf.push_at < buf.max_rows) {
                                            buf.max_rows = buf.push_at;
                                            buf.fullyBuffered = true;
                                        }

                                        // The stream hit the end, if this is the first
                                        // time set total_length
                                        if (buf.total_length <= 0) {
                                            buf.total_length = buf.push_at;
                                        }

                                        // if we need to keep reading from the stream
                                        // reset it
                                        if(buf.seekable && !(buf.fullyBuffered)){
                                            buf.inputStream->clear();
                                            buf.inputStream->seekg(0);
                                            assert(buf.inputStream->good());
                                            buf.seekData(*buf.inputStream, buf);
                                        }
                                    }
                            }

                        }
                    }
                }

                    // Now try to do an output
                {
                    LockT lock(this->mOutputs, std::defer_lock);

                    // Try to do an input
                    if ( lock.try_lock() ) {
                        for (auto p : this->outputs) {
                            OutputBuffer& buf = *p.second;
                            buf.drain();
                        }
                    }
                }


                std::chrono::milliseconds wait_duration( 1 );
                std::this_thread::sleep_for( wait_duration );


            }

            /*
             * The I/O loop is done, now flush the remaining outputs
            */
            for (auto p : this->outputs) {
                OutputBuffer& buf = *p.second;
                buf.drain();
            }

        }

        virtual iStream openReader(std::string key, bool * is_binary,
                Resetter * reset) {
            KeyParts kp = this->parseKey(key);
            *is_binary = kp.binary;

            auto kf = this->seekToData.find(kp.format);
            if (kf == this->seekToData.end()) {
                *reset = this->seekToData["dat"];
            } else {
                *reset = kf->second;
            }

            if (kp.protocol.compare("file") == 0) {
                auto fp = new std::ifstream(
                        kp.path,
                        kp.binary ?
                        std::ifstream::in | std::ifstream::binary : std::ifstream::in);

                if (fp->is_open()) {
                    return iStream(fp);
                } else {
                    std::cerr << "Unable to open " << key << " for reading..." << std::endl;
                    delete fp;
                }
            } else{
                std::cerr << "Sorry non-file protocols like "
                          << kp.protocol
                          << " are not supported yet."
                          << std::endl;
                exit(-1);
            }

            return iStream(NULL);
        }

        virtual oStream openWriter(std::string key, bool * is_binary) {
            KeyParts kp = this->parseKey(key);
            *is_binary = kp.binary;
            if (kp.protocol.compare("file") == 0) {
                auto mode = std::ofstream::out;
                if(kp.binary){ mode |= std::ofstream::binary; }
                auto fp = new std::ofstream( kp.path, mode );

                if (fp->is_open()) {
                    return oStream(fp);
                } else {
                    std::cerr << "Unable to open " << key << " for writing..." << std::endl;
                    delete fp;
                }

            } else {
                // Other other protocols later...
            }

            return oStream(NULL);
        }

        spOutput getOutput(std::string key, size_t max_rows = 0) {
            spOutput buf(NULL);
            auto iter = this->outputs.find(key);

            if (iter == this->outputs.end()) {
                LockT lock(this->mOutputs);

                bool binary = false;
                KeyParts kp = this->parseKey(key);
                oStream stream = this->openWriter(key, &binary);

                if (stream == NULL) {
                    return nullptr;
                } else {
                    assert(stream->good());
                    if (max_rows == 0)
                        max_rows = this->buffer_len;

                    if(binary){
                        buf = spOutput(new ZBinOutputBuffer(stream, max_rows));
                    }else if( kp.format.compare("dat") == 0 ){
                        buf = spOutput( new SVMOutputBuffer(stream, max_rows) );
                    } else if( kp.format.compare("arff") == 0 ){
                        buf = spOutput( new SparseArffOutputBuffer(stream, max_rows) );
                    }else{
                        buf = spOutput( new CSVOutputBuffer(stream, max_rows) );
                    }

                    this->outputs.insert(std::pair<std::string, spOutput>(key, buf));
                }
            } else {
                buf = iter->second;
            }

            return buf;
        }

        spInput getInput(std::string key, size_t max_rows = 0) {
            spInput buf(NULL);
            KeyParts kp = this->parseKey(key);
            auto iter = this->inputs.find(key);

            if (iter == this->inputs.end()) {
                LockT lock(this->mInputs);

                Resetter reset;
                bool seekable = true;
                bool binary = false;
                iStream stream = this->openReader(key, &binary, &reset);

                if (stream == NULL) {
                    return nullptr;
                } else {

                    if (max_rows == 0)
                        max_rows = this->buffer_len;

                    if (binary) {

                        buf = spInput(
                                new BinaryInputBuffer(stream, seekable, max_rows, reset));

                    } else if (kp.format.compare("xarff") == 0) {

                        buf = spInput(
                                new SparseArffInputBuffer(stream, seekable, max_rows, reset));

                    } else if (kp.format.compare("csv") == 0) {
                        buf = spInput(
                                new StringInputBuffer(stream, seekable, ',', max_rows, reset));
                    } else {
                        // This is basically CSV (which includes ARFF)
                        buf = spInput(
                                new SVMLightInputBuffer(stream, seekable, max_rows, reset));
                    }

                    if (reset != NULL) {
                        reset(*stream, *buf);
                    }

                    if (!stream->good()) {
                        std::cerr << "DataIO could not open " << key << " for reading...";
                        exit(-1);
                    }

                    this->inputs.insert(std::pair<std::string, spInput>(key, buf));
                }
            } else {
                buf = iter->second;
            }

            return buf;
        }

        /**
         * @brief parseKey
         * @param key In the format "protocol:path.format"
         * @return
         */
        KeyParts parseKey(std::string& key) {
            KeyParts ret = {"file", "", "csv"};
            size_t path_start, format_start;

            if ((path_start = key.find("::")) == key.npos) {
                path_start = 0;
            } else {
                ret.protocol.assign(key.begin(), key.begin() + path_start);
            }

            if ((format_start = key.rfind('.', key.npos)) == key.npos) {
                format_start = key.length();
            } else {
                ret.format.assign(key.begin() + format_start + 1, key.end());
            }

            ret.path.assign(key.begin() + path_start, key.end());

            ret.binary = ret.format.compare("zbin") == 0;

            return ret;
        }

        std::map<std::string, Resetter> seekToData;

        std::map<std::string, spInput> inputs;
        std::map<std::string, spOutput> outputs;
        std::mutex mInputs;
        std::mutex mOutputs;
        std::mutex mRequests;

        std::condition_variable cIORequested;

        std::vector<std::thread> threads;

        std::atomic_bool running;
        std::atomic_int request_count;

        std::atomic_int _width;

        size_t buffer_len;

    public:

        DataIO()
        : running(false),
        request_count(0),
        buffer_len(100) {

            typedef std::pair<std::string, Resetter> P;
            this->seekToData.insert(P("arff", [](std::istream& str, InputBuffer & buf) {
                size_t i = 0;
                std::string line = "";
                        const std::string attr = "@attribute";
                        str.seekg(0, str.beg);
                        assert(str.good());
                do {

                    line.assign("");
                            std::getline(str, line);

                    if (line.compare(0, attr.length(), attr) == 0) {
                        i++;
                    }

                } while (str.good()
                        && line.compare("@DATA") != 0
                        && line.compare("@data") != 0);

                        buf.set_width(i);

                        assert(str.good());
                }));

            this->seekToData.insert(P("xarff", this->seekToData["arff"]));

            this->seekToData.insert(
                    P("bin", [](std::istream& str, InputBuffer & buf) {
                        str.seekg(0, str.beg); }));

            this->seekToData.insert(
                    P("zbin", [](std::istream& str, InputBuffer & buf) {
                        str.seekg(0, str.beg); }));

            this->seekToData.insert(P("csv", [](std::istream& str, InputBuffer & buf) {
                // Expect a header on the CSV files
                std::string line;
                std::getline(str, line);
            }));

            this->seekToData.insert(
                    P("dat", [](std::istream& str, InputBuffer & buf) {

                        std::string line = "";
                        str.seekg(0, str.beg);
                        std::getline(str, line);

                        size_t start = line.find_first_not_of("# \t");

                        if( start != line.npos){
                            size_t end = line.find("\t ", start + 1);
                            buf.set_width(std::atoi(line.substr(start, end - start).c_str()));
                        }
                    }));
        }

        virtual ~DataIO() {
            this->stop();
        }

        void set_buffer_len(size_t to) {
            this->buffer_len = to;
        }

        void start(size_t nthreads = 1) {
            this->running = true;
            this->threads.resize(nthreads);
            for (size_t i = 0; i < nthreads; i++) {
                this->threads[i] = std::thread(
                        [this] {
                            this->run_thread();
                        });
            }
        }

        void stop() {
            this->running = false;
            for (std::thread& t : this->threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
        }

        void flush() {
            this->stop();
            this->start(this->threads.size());
        }

        size_t width(std::string key) {
            auto buf = this->getInput(key);
            return buf->width;
        }

        /**
         * @brief getFiller
         * @param key
         * @param max_rows
         * @return Returns a lambda function which will fill a matrix with data
         */
        Filler getFiller(std::string key, size_t max_rows = 0) {

            if (max_rows == 0)
                max_rows = this->buffer_len;

            if (key.length() > 0) {
                spInput buf = this->getInput(key, max_rows);
                assert(buf.get() != NULL);

                return [buf](OCLMatrix<T>& mat) -> bool {

                    if( buf->cycled ){
                        buf->cycled = false;
                        return false;
                    }

                    mat.getHost();
                    for ( size_t i = 0; (i < mat.rows()) && !buf->cycled; i++ ) {
                        buf->cycled = buf->getRow( mat.row_ref(i), buf->total_length );
                    }

                    return true;
                };
            }

            return nullptr;
        }

        /**
         * @brief getDrainer
         * @param key
         * @param max_rows
         * @return Returns a lambda function which will output matrix from a matrix
         */
        Drainer getDrainer(std::string key, size_t max_rows = 0) {

            if (max_rows == 0){
                max_rows = this->buffer_len;
            }

            if (key.length() > 0) {
                spOutput buf = this->getOutput(key, max_rows);
                assert(buf.get() != NULL);

                return [buf](OCLMatrix<T>& mat) {
                    buf->set_width( mat.cols() );
                    mat.getHostConst();
                    for (size_t i = 0; i < mat.rows(); i++) {
                        buf->putRow( mat.row_ref(i), mat.cols() );
                    }
                };

            }

            return nullptr;
        }
    };
}
#endif
