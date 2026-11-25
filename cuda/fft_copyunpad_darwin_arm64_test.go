//go:build darwin && arm64 && cgo

package cuda

import "testing"

func TestMetalLargeInverseFFTThenCopyUnPad(t *testing.T) {
	if testing.Short() {
		t.Skip("large 3D Metal FFT")
	}
	// On Apple M4 this inverse FFT makes MPSGraph commit and replace its
	// underlying command buffer before the following kernel is encoded.
	paddedSize := [3]int{400, 400, 200}
	dstSize := [3]int{400, 400, 100}
	plan := newFFT3DC2R(paddedSize[X], paddedSize[Y], paddedSize[Z])
	defer plan.Free()
	nx, ny, nz := plan.InputSizeFloats()
	spectrum := NewSlice(1, [3]int{nx, ny, nz})
	defer spectrum.Free()
	padded := NewSlice(1, paddedSize)
	defer padded.Free()
	dst := NewSlice(1, dstSize)
	defer dst.Free()

	// A unit DC coefficient has an unnormalized inverse of one everywhere.
	SetElem(spectrum, 0, 0, 1)
	plan.ExecAsync(spectrum, padded)
	copyUnPad(dst, padded, dstSize, paddedSize)
	for i, got := range dst.HostCopy().Host()[0] {
		if got != 1 {
			t.Fatalf("copyUnPad[%d] = %g, want 1", i, got)
		}
	}
}
