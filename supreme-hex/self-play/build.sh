cargo build --release
cp target/release/libself_play.dylib ../self_play.so
# find . -path "./target/release/*/out/libtensorflow_framework.1.dylib" -exec cp {} ../libtensorflow_framework.1.dylib \;