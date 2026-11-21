package main

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// maxDownload bounds any single extracted file (a guard against decompression bombs).
const maxDownload = 512 << 20

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func get(url string) (io.ReadCloser, error) {
	resp, err := http.Get(url) //nolint:gosec,noctx // fixed benchmark URLs
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("%s: %s", url, resp.Status)
	}
	return resp.Body, nil
}

func writeFile(dst string, r io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return err
	}
	f, err := os.Create(dst) //nolint:gosec // dst is built from the -dir flag and fixed names
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, io.LimitReader(r, maxDownload)); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func download(url, dst string) error {
	if exists(dst) {
		return nil
	}
	body, err := get(url)
	if err != nil {
		return err
	}
	defer func() { _ = body.Close() }()
	return writeFile(dst, body)
}

// fetchMassive streams the MASSIVE release and keeps only the English file.
func fetchMassive(dir string) error {
	dst := filepath.Join(dir, "massive", "en-US.jsonl")
	if exists(dst) {
		return nil
	}
	body, err := get(massiveURL)
	if err != nil {
		return err
	}
	defer func() { _ = body.Close() }()
	gz, err := gzip.NewReader(body)
	if err != nil {
		return err
	}
	defer func() { _ = gz.Close() }()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return errors.New("massive: en-US.jsonl not found in archive")
		}
		if err != nil {
			return err
		}
		if filepath.Base(h.Name) == "en-US.jsonl" {
			return writeFile(dst, tr)
		}
	}
}

func fetch(o *options) error {
	for _, b := range selected(o.only) {
		for _, f := range b.files {
			if err := download(f.url, filepath.Join(o.dir, f.dst)); err != nil {
				return err
			}
		}
		if b.name == "MASSIVE" {
			if err := fetchMassive(o.dir); err != nil {
				return err
			}
		}
	}
	fmt.Println("datasets ready in", o.dir)
	return nil
}
