// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "aabb.h"
#include "core.h"
#include "shape.h"

#include "box3d/collision.h"
#include "box3d/constants.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

/*
	Convention

	index = x + countX * ( y + countY * z )
	solid = bits[index >> 3] & ( 1 << ( index & 7 ) )

	Every voxel has 6 faces and every face has 2 triangles.
	triangleIndex = 12 * index + 2 * face + sub

	Faces are ordered -x, +x, -y, +y, -z, +z. Face corners are counter-clockwise viewed from
	outside. Corner 0 is shared by both triangles.

	   3 ---- 2
	   |  1 / |
	   |  / 0 |
	   0 ---- 1

	Triangle 0 is corners 0, 1, 2 and triangle 1 is corners 0, 2, 3.
	The diagonal is edge 3 of triangle 0 and edge 1 of triangle 1.

	Vertex indices are grid corner indices shared by adjacent faces and voxels:
	corner = x + ( countX + 1 ) * ( y + ( countY + 1 ) * z )

	A face of a solid voxel is exposed when the voxel across it is empty. Voxels outside the
	field are empty. With a border, the outer layer of voxels is never exposed but still hides
	the faces of its neighbors.
*/

#define B3_VOXEL_FACE_COUNT 6
#define B3_TRIANGLES_PER_VOXEL 12

// Corner offsets for each face, counter-clockwise viewed from outside
static const int b3_faceCorners[B3_VOXEL_FACE_COUNT][4][3] = {
	{ { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 } }, // -x
	{ { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 }, { 1, 0, 1 } }, // +x
	{ { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 }, { 0, 0, 1 } }, // -y
	{ { 0, 1, 0 }, { 0, 1, 1 }, { 1, 1, 1 }, { 1, 1, 0 } }, // +y
	{ { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } }, // -z
	{ { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } }, // +z
};

// Outward normal of each face as a voxel offset
static const int b3_faceNormals[B3_VOXEL_FACE_COUNT][3] = {
	{ -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 },
};

// The voxel across face edge k, which joins corners k and k + 1, as an offset from the voxel.
// Derived from b3_faceCorners: the offset is the sign of 2 * ( c[k] + c[k + 1] ) - sum( c ) per axis.
static const int b3_faceEdgeNeighbors[B3_VOXEL_FACE_COUNT][4][3] = {
	{ { 0, -1, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 0, -1 } }, // -x
	{ { 0, 0, -1 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, -1, 0 } }, // +x
	{ { 0, 0, -1 }, { 1, 0, 0 }, { 0, 0, 1 }, { -1, 0, 0 } }, // -y
	{ { -1, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0, -1 } }, // +y
	{ { -1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, -1, 0 } }, // -z
	{ { 0, -1, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { -1, 0, 0 } }, // +z
};

static inline int b3GetVoxelIndex( const b3VoxelFieldData* field, int x, int y, int z )
{
	return x + field->countX * ( y + field->countY * z );
}

static inline int b3GetVoxelCornerIndex( const b3VoxelFieldData* field, int x, int y, int z )
{
	return x + ( field->countX + 1 ) * ( y + ( field->countY + 1 ) * z );
}

static inline b3Vec3 b3GetVoxelCorner( const b3VoxelFieldData* field, int x, int y, int z )
{
	return b3Mul( field->scale, (b3Vec3){ (float)x, (float)y, (float)z } );
}

static inline int b3GetVoxelBorder( const b3VoxelFieldData* field )
{
	return field->hasBorder ? 1 : 0;
}

typedef struct b3VoxelRange
{
	int x1, y1, z1;
	int x2, y2, z2;
} b3VoxelRange;

// Convert a local coordinate to a voxel coordinate. Clamped in float first so that huge bounds
// cannot overflow the integer conversion. The result is at most one voxel outside the field.
static inline int b3GetVoxelCoordinate( float value, float scale, int count )
{
	return (int)floorf( b3ClampFloat( value / scale, -1.0f, (float)count ) );
}

// The interior voxel range overlapping the local bounds. Empty when x1 > x2 on any axis.
static b3VoxelRange b3GetVoxelRange( const b3VoxelFieldData* field, b3AABB bounds )
{
	int border = b3GetVoxelBorder( field );
	b3Vec3 scale = field->scale;
	b3Vec3 lower = bounds.lowerBound;
	b3Vec3 upper = bounds.upperBound;

	b3VoxelRange range;
	range.x1 = b3MaxInt( b3GetVoxelCoordinate( lower.x, scale.x, field->countX ), border );
	range.y1 = b3MaxInt( b3GetVoxelCoordinate( lower.y, scale.y, field->countY ), border );
	range.z1 = b3MaxInt( b3GetVoxelCoordinate( lower.z, scale.z, field->countZ ), border );
	range.x2 = b3MinInt( b3GetVoxelCoordinate( upper.x, scale.x, field->countX ), field->countX - 1 - border );
	range.y2 = b3MinInt( b3GetVoxelCoordinate( upper.y, scale.y, field->countY ), field->countY - 1 - border );
	range.z2 = b3MinInt( b3GetVoxelCoordinate( upper.z, scale.z, field->countZ ), field->countZ - 1 - border );
	return range;
}

// A face is exposed when the voxel across it is empty
static inline bool b3IsVoxelFaceExposed( const b3VoxelFieldData* field, int x, int y, int z, int face )
{
	const int* n = b3_faceNormals[face];
	return b3IsVoxelSolid( field, x + n[0], y + n[1], z + n[2] ) == false;
}

// The four corners of a face in local space, counter-clockwise viewed from outside
static inline void b3GetVoxelFaceCorners( const b3VoxelFieldData* field, int x, int y, int z, int face, b3Vec3 corners[4] )
{
	for ( int i = 0; i < 4; ++i )
	{
		const int* offset = b3_faceCorners[face][i];
		corners[i] = b3GetVoxelCorner( field, x + offset[0], y + offset[1], z + offset[2] );
	}
}

b3VoxelFieldData* b3CreateVoxelField( const b3VoxelFieldDef* def )
{
	B3_ASSERT( def->voxels != NULL );
	B3_ASSERT( def->countX > 0 && def->countY > 0 && def->countZ > 0 );
	B3_ASSERT( def->scale.x > 0.0f && def->scale.y > 0.0f && def->scale.z > 0.0f );

	int border = def->hasBorder ? 1 : 0;
	B3_ASSERT( def->countX > 2 * border && def->countY > 2 * border && def->countZ > 2 * border );

	int64_t voxelCount = (int64_t)def->countX * def->countY * def->countZ;
	int64_t cornerCount = (int64_t)( def->countX + 1 ) * ( def->countY + 1 ) * ( def->countZ + 1 );

	// Triangle and corner indices must fit in an int
	B3_ASSERT( B3_TRIANGLES_PER_VOXEL * voxelCount < INT32_MAX );
	B3_ASSERT( cornerCount < INT32_MAX );
	B3_UNUSED( cornerCount );

	int bitByteCount = (int)( ( voxelCount + 7 ) / 8 );

	// Single blob: struct followed by the bit and material arrays. Layout mirrors
	// b3HeightFieldData so the recording path can copy it with one memcpy.
	size_t byteCount = b3AlignUp8( sizeof( b3VoxelFieldData ) );
	int bitsOffset = (int)byteCount;
	byteCount += b3AlignUp8( bitByteCount );

	int materialOffset = 0;
	if ( def->materialIndices != NULL )
	{
		materialOffset = (int)byteCount;
		byteCount += b3AlignUp8( (size_t)voxelCount );
	}

	// Zero the whole blob so alignment padding is defined. The construction-time hash
	// sweeps raw bytes and would otherwise pick up uninitialized padding.
	b3VoxelFieldData* field = (b3VoxelFieldData*)b3Alloc( byteCount );
	memset( field, 0, byteCount );

	field->version = B3_VOXEL_FIELD_VERSION;
	field->byteCount = (int)byteCount;
	field->scale = def->scale;
	field->countX = def->countX;
	field->countY = def->countY;
	field->countZ = def->countZ;
	field->bitsOffset = bitsOffset;
	field->materialOffset = materialOffset;
	field->hasBorder = (uint8_t)border;

	// The border is excluded from the bounds
	b3Vec3 scale = def->scale;
	field->aabb.lowerBound = b3MulSV( (float)border, scale );
	field->aabb.upperBound = b3Mul(
		scale, (b3Vec3){ (float)( def->countX - border ), (float)( def->countY - border ), (float)( def->countZ - border ) } );

	uint8_t* bits = (uint8_t*)( (intptr_t)field + bitsOffset );
	int solidCount = 0;
	for ( int z = 0; z < def->countZ; ++z )
	{
		bool interiorZ = border <= z && z < def->countZ - border;
		for ( int y = 0; y < def->countY; ++y )
		{
			bool interiorY = border <= y && y < def->countY - border;
			for ( int x = 0; x < def->countX; ++x )
			{
				int index = x + def->countX * ( y + def->countY * z );
				if ( def->voxels[index] != 0 )
				{
					bits[index >> 3] |= (uint8_t)( 1 << ( index & 7 ) );

					bool interiorX = border <= x && x < def->countX - border;
					if ( interiorX && interiorY && interiorZ )
					{
						solidCount += 1;
					}
				}
			}
		}
	}

	field->solidCount = solidCount;

	if ( materialOffset != 0 )
	{
		uint8_t* materialIndices = (uint8_t*)( (intptr_t)field + materialOffset );
		memcpy( materialIndices, def->materialIndices, (size_t)voxelCount );
	}

	field->hash = 0;
	field->hash = b3Hash64NonZero( (const uint8_t*)field, field->byteCount );

	return field;
}

b3VoxelFieldData* b3CreateVoxelWave( int countX, int countY, int countZ, int offsetX, int offsetZ, b3Vec3 scale, float frequencyX,
									 float frequencyZ, bool hasBorder )
{
	int border = hasBorder ? 1 : 0;
	int voxelCount = countX * countY * countZ;
	uint8_t* voxels = (uint8_t*)b3Alloc( voxelCount );

	for ( int z = 0; z < countZ; ++z )
	{
		// Sample the wave in tile space so adjacent tiles agree in their borders
		float gz = (float)( z - border + offsetZ );
		b3CosSin cz = b3ComputeCosSin( frequencyZ * gz );

		for ( int x = 0; x < countX; ++x )
		{
			float gx = (float)( x - border + offsetX );
			b3CosSin cx = b3ComputeCosSin( frequencyX * gx );

			float height = (float)countY * ( 0.5f + 0.2f * cx.sine + 0.2f * cz.cosine );

			for ( int y = 0; y < countY; ++y )
			{
				voxels[x + countX * ( y + countY * z )] = (float)y < height ? 1 : 0;
			}
		}
	}

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.scale = scale;
	def.countX = countX;
	def.countY = countY;
	def.countZ = countZ;
	def.hasBorder = hasBorder;

	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	b3Free( voxels, voxelCount );

	return field;
}

void b3DestroyVoxelField( b3VoxelFieldData* field )
{
	b3Free( field, field->byteCount );
}

b3AABB b3ComputeVoxelFieldAABB( const b3VoxelFieldData* shape, b3Transform transform )
{
	return b3AABB_Transform( transform, shape->aabb );
}

// Flags for one outer edge of a face. Looking across the edge the surface either steps down
// (convex), continues flat, or rises into a wall (concave).
static int b3GetVoxelEdgeFlags( const b3VoxelFieldData* field, int x, int y, int z, int face, int edge, int concaveBit,
								int inverseBit )
{
	const int* n = b3_faceNormals[face];
	const int* d = b3_faceEdgeNeighbors[face][edge];

	bool sideSolid = b3IsVoxelSolid( field, x + d[0], y + d[1], z + d[2] );
	if ( sideSolid == false )
	{
		return inverseBit;
	}

	bool aboveSolid = b3IsVoxelSolid( field, x + d[0] + n[0], y + d[1] + n[1], z + d[2] + n[2] );
	if ( aboveSolid )
	{
		return concaveBit;
	}

	return concaveBit | inverseBit;
}

int b3GetVoxelFieldTriangleCount( const b3VoxelFieldData* field )
{
	return B3_TRIANGLES_PER_VOXEL * field->countX * field->countY * field->countZ;
}

b3Triangle b3GetVoxelFieldTriangle( const b3VoxelFieldData* field, int triangleIndex )
{
	B3_ASSERT( 0 <= triangleIndex && triangleIndex < b3GetVoxelFieldTriangleCount( field ) );

	int voxelIndex = triangleIndex / B3_TRIANGLES_PER_VOXEL;
	int face = ( triangleIndex - B3_TRIANGLES_PER_VOXEL * voxelIndex ) >> 1;
	int sub = triangleIndex & 1;

	int countX = field->countX;
	int countY = field->countY;
	int x = voxelIndex % countX;
	int y = ( voxelIndex / countX ) % countY;
	int z = voxelIndex / ( countX * countY );

	B3_ASSERT( b3IsVoxelSolid( field, x, y, z ) );

	// Corners 0, 1, 2 or 0, 2, 3
	int corners[3] = { 0, 1 + sub, 2 + sub };
	int indices[3];

	b3Triangle triangle;
	for ( int i = 0; i < 3; ++i )
	{
		const int* offset = b3_faceCorners[face][corners[i]];
		int cx = x + offset[0];
		int cy = y + offset[1];
		int cz = z + offset[2];
		triangle.vertices[i] = b3GetVoxelCorner( field, cx, cy, cz );
		indices[i] = b3GetVoxelCornerIndex( field, cx, cy, cz );
	}

	triangle.i1 = indices[0];
	triangle.i2 = indices[1];
	triangle.i3 = indices[2];

	if ( sub == 0 )
	{
		// Edge 1 is face edge 0, edge 2 is face edge 1, edge 3 is the diagonal
		triangle.flags = b3GetVoxelEdgeFlags( field, x, y, z, face, 0, b3_concaveEdge1, b3_inverseConcaveEdge1 ) |
						 b3GetVoxelEdgeFlags( field, x, y, z, face, 1, b3_concaveEdge2, b3_inverseConcaveEdge2 ) | b3_flatEdge3;
	}
	else
	{
		// Edge 1 is the diagonal, edge 2 is face edge 2, edge 3 is face edge 3
		triangle.flags = b3_flatEdge1 | b3GetVoxelEdgeFlags( field, x, y, z, face, 2, b3_concaveEdge2, b3_inverseConcaveEdge2 ) |
						 b3GetVoxelEdgeFlags( field, x, y, z, face, 3, b3_concaveEdge3, b3_inverseConcaveEdge3 );
	}

	return triangle;
}

int b3GetVoxelFieldMaterial( const b3VoxelFieldData* field, int triangleIndex )
{
	B3_ASSERT( 0 <= triangleIndex && triangleIndex < b3GetVoxelFieldTriangleCount( field ) );

	const uint8_t* materialIndices = b3GetVoxelFieldMaterialIndices( field );
	if ( materialIndices == NULL )
	{
		return 0;
	}

	return materialIndices[triangleIndex / B3_TRIANGLES_PER_VOXEL];
}

void b3QueryVoxelField( const b3VoxelFieldData* field, b3AABB bounds, b3MeshQueryFcn* fcn, void* context )
{
	b3VoxelRange range = b3GetVoxelRange( field, bounds );

	// Outer loop on z, then y, then x so that triangle indices increase monotonically.
	for ( int z = range.z1; z <= range.z2; ++z )
	{
		for ( int y = range.y1; y <= range.y2; ++y )
		{
			for ( int x = range.x1; x <= range.x2; ++x )
			{
				if ( b3IsVoxelSolid( field, x, y, z ) == false )
				{
					continue;
				}

				int voxelIndex = b3GetVoxelIndex( field, x, y, z );

				for ( int face = 0; face < B3_VOXEL_FACE_COUNT; ++face )
				{
					if ( b3IsVoxelFaceExposed( field, x, y, z, face ) == false )
					{
						continue;
					}

					b3Vec3 corners[4];
					b3GetVoxelFaceCorners( field, x, y, z, face, corners );
					int triangleIndex = B3_TRIANGLES_PER_VOXEL * voxelIndex + 2 * face;

					if ( fcn( corners[0], corners[1], corners[2], triangleIndex, context ) == false )
					{
						return;
					}

					if ( fcn( corners[0], corners[2], corners[3], triangleIndex + 1, context ) == false )
					{
						return;
					}
				}
			}
		}
	}
}
