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
